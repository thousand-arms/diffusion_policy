"""Dataset for the edge-tracing task.

Consumes the zarr replay buffer produced by
/home/andrew/Dev/stereo_inertial_data_collection (INDEMIND stereo-grayscale
camera + ORB-SLAM3 6-DoF pose tracking) with layout:

    data/cam0                (T, H, W, 1) uint8   grayscale, fisheye-distorted
    data/cam1                (T, H, W, 1) uint8   grayscale, fisheye-distorted
    data/cam_pos             (T, 3)       float32 table-frame position (meters)
    data/cam_rot_axis_angle  (T, 3)       float32 table-frame rotation-vector (radians)
    meta/episode_ends        (E,)         int64

Produces samples whose pose observations and action targets are both
expressed in a single relative frame anchored at the last observation step
(UMI convention — see `memory/project_umi_relative_pose_convention.md`).
"""

from typing import Dict
import copy
import os
import pathlib
import shutil
from datetime import datetime

import numpy as np
import torch
import zarr
from filelock import FileLock
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat,
)


def _build_pos_rot6d_action_normalizer(action_arr: np.ndarray) -> SingleFieldLinearNormalizer:
    """Build a 9D action normalizer that range-normalizes the first 3 dims
    (position) and leaves the last 6 dims (6D rotation) untouched.

    6D rotation components are already bounded in [-1, 1] by construction
    (two orthonormal columns of a rotation matrix); rescaling them would
    break that geometric property and produce outputs that no longer decode
    to valid rotations.
    """
    assert action_arr.ndim == 2 and action_arr.shape[-1] == 9
    pos_stats = array_to_stats(action_arr[:, :3])
    rot_stats = array_to_stats(action_arr[:, 3:])

    pos_norm = get_range_normalizer_from_stat(pos_stats)
    rot_norm = get_identity_normalizer_from_stat(rot_stats)

    # Concatenate scale / offset / input_stats from both sub-normalizers into
    # a single 9-D SingleFieldLinearNormalizer so the policy can normalize
    # the action in one shot via `normalizer['action'].normalize(...)`.
    scale = torch.cat([
        pos_norm.params_dict['scale'],
        rot_norm.params_dict['scale'],
    ], dim=0)
    offset = torch.cat([
        pos_norm.params_dict['offset'],
        rot_norm.params_dict['offset'],
    ], dim=0)

    combined_stats = {}
    for k in ['min', 'max', 'mean', 'std']:
        combined_stats[k] = torch.cat([
            pos_norm.params_dict['input_stats'][k],
            rot_norm.params_dict['input_stats'][k],
        ], dim=0)

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=combined_stats)


def _load_replay_buffer_via_lmdb_cache(
        zarr_path: str,
        cache_dir: str,
        ) -> ReplayBuffer:
    """Load the source .zarr.zip into an LMDB-backed `ReplayBuffer`.

    Two-phase:
      1) Cache build (first call for a given source mtime): acquire a
         FileLock, copy `zarr_path` into `<cache_dir>/<stem>_<mtime>.zarr.mdb`
         via `ReplayBuffer.copy_from_store`, release the lock.
      2) Cache open (every call): open the LMDB store read-only + mmap'd
         (`lock=False` so multiple DataLoader worker processes can share
         the inherited mmap without LMDB's writer lock fighting them).

    A failed cache build is rolled back by `shutil.rmtree` on the
    half-written LMDB directory so the next run sees a clean miss and
    retries, rather than opening a corrupt db.

    Cache path is keyed by the source zarr's mtime so regenerating the
    source guarantees a cache miss and rebuild — no filename-reuse
    staleness bug.
    """
    zarr_path_abs = os.path.expanduser(zarr_path)
    cache_dir_path = pathlib.Path(os.path.expanduser(cache_dir))
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(zarr_path_abs):
        raise FileNotFoundError(
            f"Source zarr archive not found: {zarr_path_abs}")

    mod_time = os.path.getmtime(zarr_path_abs)
    stamp = datetime.fromtimestamp(mod_time).isoformat(timespec='seconds')
    stem = os.path.basename(zarr_path_abs).split('.')[0]
    cache_stem = f'{stem}_{stamp}'
    # zarr.LMDBStore defaults to subdir=True via python-lmdb, so
    # cache_path is a *directory* containing data.mdb / lock.mdb. Match
    # UMI's naming scheme (`.zarr.mdb` suffix on the directory) so that
    # `shutil.rmtree` on failure is the right cleanup op.
    cache_path = cache_dir_path / (cache_stem + '.zarr.mdb')
    lock_path = cache_dir_path / (cache_stem + '.lock')

    with FileLock(str(lock_path)):
        if not cache_path.exists():
            print(f'[TraceDataset] cache miss, building: {cache_path}')
            try:
                # writemap=True gives LMDB a writable mmap during the copy,
                # which is the fastest way to populate a fresh store.
                # metasync/sync/map_async=False skip fsync flushes — we
                # are writing a pure cache derived from the source zip, so
                # crash-durability is not required: on a crash the partial
                # cache is rmtree'd and rebuilt from the source.
                with zarr.LMDBStore(
                        str(cache_path),
                        writemap=True, metasync=False,
                        sync=False, map_async=True, lock=False,
                        ) as lmdb_store:
                    with zarr.ZipStore(zarr_path_abs, mode='r') as zip_store:
                        ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=lmdb_store)
                print(f'[TraceDataset] cache built: {cache_path}')
            except BaseException:
                # Half-written cache -> delete so the next run rebuilds
                # cleanly. Catch BaseException to also handle
                # KeyboardInterrupt mid-copy.
                if cache_path.exists():
                    shutil.rmtree(cache_path, ignore_errors=True)
                raise
        else:
            print(f'[TraceDataset] cache hit: {cache_path}')

    # Quick sanity/footprint report so stale caches in the dir are obvious
    # at a glance. Uses `st_blocks * 512` (= bytes actually allocated on
    # disk, same as `du`) instead of `st_size`, because LMDB opens
    # data.mdb with `map_size=2**40` and ftruncates it to that apparent
    # size — so `st_size` would report ~1 TiB per cache entry, which is a
    # sparse-file illusion that would make these logs look alarming.
    try:
        entries = sorted(cache_dir_path.glob('*.zarr.mdb'))
        total_bytes = 0
        for entry in entries:
            for f in entry.rglob('*'):
                if f.is_file():
                    total_bytes += f.stat().st_blocks * 512
        print(f'[TraceDataset] cache dir {cache_dir_path}: '
              f'{len(entries)} entries, {total_bytes / 1e9:.2f} GB on disk')
    except OSError:
        pass

    # Open the cache read-only. lock=False disables LMDB's writer lock
    # (safe because nothing writes) so DataLoader workers forked off this
    # process all share the same mmap without mutex contention.
    store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
    return ReplayBuffer.create_from_group(group=zarr.group(store))


class TraceDataset(BaseImageDataset):
    def __init__(self,
            zarr_path: str,
            cache_dir: str,
            horizon: int = 16,
            n_obs_steps: int = 2,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: int = None,
            image_normalization: str = 'range',
            ):
        """
        Args:
            zarr_path: path to the source replay buffer (.zarr.zip).
                This is the archive format written by the data-collection
                repo. It is *not* read directly at training time — see
                `cache_dir` below.
            cache_dir: directory in which to maintain an LMDB-backed cache
                of the source zarr. On first construction the source
                `zarr_path` is copied into
                `<cache_dir>/<zarr_stem>_<mtime>.zarr.mdb` (guarded by a
                `FileLock`); subsequent constructions with the same source
                mtime hit the cache and skip the copy. The cache is then
                opened read-only and memory-mapped, so DataLoader worker
                processes share the OS page cache instead of fork-CoW'ing
                a numpy buffer per worker. RAM cost decouples from total
                dataset size, which is what lets us scale past ~RAM-sized
                datasets and run large batches without OOM.

                ZipStore is fine for archival but bad for training access:
                it is append-only, chunks can't be mmap'd, and concurrent
                workers fight over a single shared file descriptor. LMDB
                is a random-access b+tree designed exactly for this.

                Cache invalidation is by source mtime encoded in the
                filename, so regenerating the source zarr auto-produces a
                new cache entry. Old entries are not auto-deleted; remove
                them manually when you want to reclaim disk.
            horizon: full sequence length returned per sample (=T in the
                paper). Obs and action share this length; the policy slices
                obs to `n_obs_steps` and action to its own window.
            n_obs_steps: number of observation steps (=To). The pose at
                index `n_obs_steps - 1` is the relative-frame anchor:
                after relativization its pose is the identity and all
                other obs / action poses are rewritten in its frame.
            pad_before, pad_after: number of virtual pre/post-episode frames
                the sampler is allowed to hang off; pre-padding is filled
                by repeating the first real frame, post-padding the last.
            seed: RNG seed for train/val split and train downsampling.
            val_ratio: fraction of episodes to hold out for validation.
            max_train_episodes: optional cap on number of training episodes
                (for data-efficiency ablations). `None` uses all of them.
            image_normalization:
                'range'    -> map [0,1] -> [-1,1] via `get_image_range_normalizer`.
                             Use for CNNs trained from scratch.
                'identity' -> pass images through unchanged in [0,1]. Use
                             when the visual backbone applies its own
                             preprocessing (e.g. ImageNet-pretrained timm).
        """
        super().__init__()
        assert image_normalization in ('range', 'identity'), \
            f"image_normalization must be 'range' or 'identity', got {image_normalization!r}"
        assert 1 <= n_obs_steps <= horizon, \
            f"n_obs_steps ({n_obs_steps}) must be in [1, horizon={horizon}]"

        self.replay_buffer = _load_replay_buffer_via_lmdb_cache(
            zarr_path=zarr_path, cache_dir=cache_dir)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_normalization = image_normalization

    # ------------------------------------------------------------------ split

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    # -------------------------------------------------------------- sampling

    def __len__(self) -> int:
        return len(self.sampler)

    def _relativize_poses(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Build homogeneous pose matrices from the sampled `cam_pos` and
        `cam_rot_axis_angle`, re-express every step in the frame of
        `pose_mat[n_obs_steps - 1]`, and decompose into position (3D) and
        6D rotation (first two columns of the rotation matrix, stacked).

        Returns a dict with:
            'cam_pos'     (T, 3) float32  relativized position
            'cam_rot_6d'  (T, 6) float32  relativized 6D rotation
            'action'      (T, 9) float32  concatenation of the above

        After relativization, `pose_mat[n_obs_steps - 1]` is the identity,
        so `cam_pos[n_obs_steps - 1]` is [0, 0, 0] and
        `cam_rot_6d[n_obs_steps - 1]` is [1, 0, 0, 0, 1, 0].
        """
        raw_pos = sample['cam_pos']                 # (T, 3) float32
        raw_rotvec = sample['cam_rot_axis_angle']   # (T, 3) float32
        T = raw_pos.shape[0]

        pose_mat = np.zeros((T, 4, 4), dtype=np.float64)
        pose_mat[:, :3, :3] = Rotation.from_rotvec(raw_rotvec).as_matrix()
        pose_mat[:, :3, 3] = raw_pos
        pose_mat[:, 3, 3] = 1.0

        anchor_inv = np.linalg.inv(pose_mat[self.n_obs_steps - 1])
        rel_pose_mat = anchor_inv @ pose_mat        # (T, 4, 4)

        rel_pos = rel_pose_mat[:, :3, 3].astype(np.float32)
        # 6D rotation = first two columns of R, stacked:
        # [r00, r10, r20, r01, r11, r21]
        rel_rot_6d = np.concatenate([
            rel_pose_mat[:, :3, 0],
            rel_pose_mat[:, :3, 1],
        ], axis=-1).astype(np.float32)

        action = np.concatenate([rel_pos, rel_rot_6d], axis=-1)

        return {
            'cam_pos': rel_pos,
            'cam_rot_6d': rel_rot_6d,
            'action': action,
        }

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict:
        rel = self._relativize_poses(sample)

        # (T, H, W, 1) uint8  ->  (T, 1, H, W) float32 in [0, 1]
        cam0 = np.moveaxis(sample['cam0'], -1, 1).astype(np.float32) / 255.0
        cam1 = np.moveaxis(sample['cam1'], -1, 1).astype(np.float32) / 255.0
        # Triplicate to (T, 3, H, W) so a stock ImageNet-pretrained
        # 3-channel ResNet can consume grayscale input without surgery
        # on its conv1. Cheap at 224x224 and keeps the door open to
        # swapping in any pretrained backbone later.
        cam0 = np.repeat(cam0, 3, axis=1)
        cam1 = np.repeat(cam1, 3, axis=1)

        return {
            'obs': {
                'cam0': cam0,                    # (T, 3, H, W)
                'cam1': cam1,                    # (T, 3, H, W)
                'cam_pos': rel['cam_pos'],       # (T, 3)
                'cam_rot_6d': rel['cam_rot_6d'], # (T, 6)
            },
            'action': rel['action'],             # (T, 9)
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

    # ---------------------------------------------------------- normalization

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Fit normalization statistics on *relativized* poses and actions.

        We cannot fit directly on the raw zarr arrays: the policy only ever
        sees poses in the per-sample relative frame, whose spatial extent is
        much smaller than the absolute table-frame poses stored on disk.
        Fitting on the raw arrays would undersize the dynamic range of the
        normalizer and give the diffusion model a hopelessly peaked target
        distribution.

        Instead, iterate the sampler once, run every window through
        `_relativize_poses`, and accumulate per-feature stats on what the
        policy will actually see. Image loading is skipped here for speed.
        """
        normalizer = LinearNormalizer()

        pos_chunks = []
        rot_chunks = []
        action_chunks = []
        for idx in tqdm(range(len(self.sampler)),
                        desc='fitting trace normalizer'):
            sample = self.sampler.sample_sequence(idx)
            rel = self._relativize_poses(sample)
            pos_chunks.append(rel['cam_pos'])
            rot_chunks.append(rel['cam_rot_6d'])
            action_chunks.append(rel['action'])

        # Each chunk is (horizon, D); concatenating along axis 0 folds the
        # window-time axis into the sample axis so stats are computed per
        # feature over every frame of every window.
        obs_pos_flat = np.concatenate(pos_chunks, axis=0)    # (N*T, 3)
        obs_rot_flat = np.concatenate(rot_chunks, axis=0)    # (N*T, 6)
        action_flat = np.concatenate(action_chunks, axis=0)  # (N*T, 9)

        # Observation poses: range on position, identity on 6D rotation.
        normalizer['cam_pos'] = get_range_normalizer_from_stat(
            array_to_stats(obs_pos_flat))
        normalizer['cam_rot_6d'] = get_identity_normalizer_from_stat(
            array_to_stats(obs_rot_flat))

        # Action: hybrid 9D normalizer (range on first 3 dims, identity on
        # last 6). Built manually because this repo lacks UMI's
        # `concatenate_normalizer` helper.
        normalizer['action'] = _build_pos_rot6d_action_normalizer(action_flat)

        # Images.
        if self.image_normalization == 'range':
            normalizer['cam0'] = get_image_range_normalizer()
            normalizer['cam1'] = get_image_range_normalizer()
        else:  # 'identity'
            normalizer['cam0'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['cam1'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer
