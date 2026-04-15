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
