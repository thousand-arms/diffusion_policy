# Project notes

## Piper robot arm

If the user runs into issues using the Piper arm (SDK behavior, Euler conventions, feedback/status fields, CAN control modes, fault codes, etc.), search the Piper SDK source for answers before speculating:

- SDK root: `/home/andrew/Dev/piper_sdk/piper_sdk`
- Main interface: `piper_sdk/interface/piper_interface_v2.py` (class `C_PiperInterface_V2`)
- Feedback message structs: `piper_sdk/piper_msgs/msg_v2/feedback/`
- Transmit message structs: `piper_sdk/piper_msgs/msg_v2/transmit/`
- Demo / example usage: `piper_sdk/demo/V2/`
- TF utilities (euler/quat conversions): `piper_sdk/utils/tf.py`

Known convention: `EndPoseCtrl` uses RX/RY/RZ as **extrinsic XYZ euler** in 0.001 deg (scipy `Rotation.from_euler("xyz", ..., degrees=True)`). Position in 0.001 mm.

## INDEMIND camera (pyindemind)

Python bindings for the INDEMIND stereo-inertial camera are in a separate repo. If `import pyindemind` fails on this machine, install with:

```
git clone https://github.com/thousand-arms/Indemind-SDK
cd Indemind-SDK/python
pip install .
```

Example usage: `/home/andrew/Dev/Indemind-SDK/python/examples/stream_stereo_imu.py`. Main API: `pyindemind.Camera()` → `.start(resolution, img_hz, imu_hz)`, `.get_frame(timeout_s)` returns `(ts, left, right)`, `.drain_imu()`, `.stop()`, `.get_module_params()`.

## Training setup

- **Task**: rectangle edge-tracing with Piper arm + INDEMIND stereo camera (50 Hz, 640x400 grayscale)
- **Model**: 100M param UNet diffusion policy, DDIM-16 inference (~130ms), frozen ResNet18 encoder (ImageNet pretrained)
- **Data**: 137 episodes in `dataset.zarr.zip`, 7 val episodes (5% split, seed=42)
- **Config**: horizon=16, n_obs_steps=2, down_sample_steps=3 (effective ~17 Hz, ~960ms per horizon), batch_size=32, num_workers=2
- **Eval**: combined val_every=20 epochs — computes val_loss (diffusion loss) + val_pos_mse (full DDIM inference on entire val set, position-only MSE). val_pos_mse is the primary metric for model quality. Train loss / diffusion loss is a poor indicator of actual prediction quality.
- **Checkpointing**: every 5 epochs for latest.ckpt + topk train_loss. topk val uses val_pos_mse (only available at val_every intervals).
- **Inference on robot**: `python -m piper.step_eval -c <checkpoint>`. Piper arm_status=0x04 (TARGET_POS_EXCEEDS_LIMIT) can be a latched error — disable/re-enable arm to clear.
- **Resume training**: must specify `hydra.run.dir=<existing_output_dir>` since output dirs are timestamped.
