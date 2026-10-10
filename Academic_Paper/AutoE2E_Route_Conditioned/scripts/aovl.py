# ruff: noqa
"""Decode AOVL overlay bodies (schema v1-v5 layout family) with numpy only."""
import gzip, hashlib, struct
import numpy as np

def sample_uid_hash(uid: str) -> int:
    d = hashlib.sha256(uid.encode()).digest()
    return int.from_bytes(d[:8], "little", signed=False)

def decode(payload: bytes):
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    assert raw[:4] == b"AOVL", raw[:4]
    ver, flags, n, s, horizon, dims, reserved = struct.unpack_from("<HHIHHHH", raw, 4)
    off = 20
    seeds = np.frombuffer(raw, dtype="<i8", count=s, offset=off); off += 8 * s
    directory = {}
    for i in range(n):
        h, row = struct.unpack_from("<QI", raw, off); off += 12
        directory[h] = row
    ctrl_len = n * s * horizon * dims
    controls = np.frombuffer(raw, dtype="<f4", count=ctrl_len, offset=off).reshape(n, s, horizon, dims); off += 4 * ctrl_len
    v0 = np.frombuffer(raw, dtype="<f4", count=n, offset=off); off += 4 * n
    hc = reserved
    scales = heatmaps = None
    if hc:
        scales = np.frombuffer(raw, dtype="<f4", count=n * hc, offset=off).reshape(n, hc); off += 4 * n * hc
        heatmaps = np.frombuffer(raw, dtype="u1", count=n * hc * 1024, offset=off).reshape(n, hc, 32, 32); off += n * hc * 1024
    assert off == len(raw), (off, len(raw))
    return dict(version=ver, flags=flags, n=n, seeds=seeds, directory=directory,
                controls=controls, v0=v0, heatmap_scales=scales, heatmaps=heatmaps)

def rollout(controls, v0, dt=0.1):
    """Semi-implicit unicycle rollout identical to training/losses/control_rollout.py."""
    a = controls[:, 0].astype(np.float64); k = controls[:, 1].astype(np.float64)
    v = float(v0); speeds = []
    for t in range(len(a)):
        v = max(v + a[t] * dt, 0.0); speeds.append(v)
    speeds = np.asarray(speeds)
    heading = np.cumsum(speeds * k * dt)
    x = np.cumsum(speeds * np.cos(heading) * dt); y = np.cumsum(speeds * np.sin(heading) * dt)
    return np.stack([x, y], 1), heading, speeds
