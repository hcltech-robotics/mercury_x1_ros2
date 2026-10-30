"""3D map from the RTAB-Map database: every map node keeps its RGB-D frame, so `rtabmap-export` can
assemble a coloured cloud (or a textured mesh) of the whole map on the loop-closure-optimised poses.

The live database is written by rtabmap while we read. Never open it with SQLite: even a read-only
connection takes a shared lock, and rtabmap aborts (FATAL "database is locked") when a write meets it.
So the snapshot is a raw byte copy (no locks), verified with PRAGMA quick_check on the copy and retried
if it caught a write half-way. Measured on the Orin Nano (office map, ~120 MB): copy ~0.3 s, cloud
1.6 s, textured mesh ~50 s.
"""
from __future__ import annotations

import glob
import os
import shutil
import sqlite3
import subprocess
import time
import zipfile

import numpy as np

EXPORT_DIR = os.path.expanduser("~/maps/exports")
SNAP_DIR = "/tmp/mx1_slam3d"
_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
              "float": "<f4", "double": "<f8"}


def snapshot(db_path: str, tries: int = 5) -> str:
    """lock-free copy of a database that rtabmap may be writing (see the module notes)"""
    os.makedirs(SNAP_DIR, exist_ok=True)
    out = os.path.join(SNAP_DIR, os.path.basename(db_path))
    for _ in range(tries):
        shutil.copyfile(db_path, out)
        for side in ("-journal", "-wal", "-shm"):     # never pair the copy with a stale journal
            if os.path.exists(out + side):
                os.remove(out + side)
        try:
            con = sqlite3.connect(out)
            ok = con.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            con.close()
        except sqlite3.DatabaseError:
            ok = False
        if ok:
            return out
        time.sleep(1.0)
    raise RuntimeError("could not take a consistent copy of the map database (rtabmap busy writing)")


def _export(db: str, out_dir: str, name: str, args: list[str], timeout: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["nice", "-n", "10", "rtabmap-export", *args, "--max_range", "4", "--output_dir", out_dir,
           "--output", name, db]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["?"]
        raise RuntimeError(f"rtabmap-export failed: {tail[0][:200]}")


def export_cloud(db_path: str, out_dir: str, name: str, voxel: float = 0.02, poses: bool = False) -> str:
    """coloured point cloud (PLY) -> path; poses=True also writes <name>_poses.txt (see read_poses)"""
    _export(snapshot(db_path), out_dir, name, ["--cloud", "--voxel", str(voxel)] + (["--poses"] if poses else []),
            timeout=300)
    return os.path.join(out_dir, f"{name}_cloud.ply")


def read_poses(path: str) -> dict[int, tuple[float, float, float]]:
    """rtabmap-export --poses (format 11: stamp x y z qx qy qz qw id) -> node id -> position"""
    out = {}
    with open(path) as f:
        for line in f:
            w = line.split()
            if len(w) >= 9 and not w[0].startswith("#"):
                out[int(w[8])] = (float(w[1]), float(w[2]), float(w[3]))
    return out


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """least-squares R, t with dst ~ R @ src + t (Kabsch), and the mean residual"""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ d @ u.T
    t = cd - r @ cs
    return r, t, float(np.linalg.norm(src @ r.T + t - dst, axis=1).mean())


def export_mesh(db_path: str, out_dir: str, name: str) -> str:
    """textured mesh (OBJ + MTL + JPG), zipped -> path of the zip"""
    _export(snapshot(db_path), out_dir, name, ["--mesh", "--texture", "--texture_size", "4096", "--voxel", "0.02"],
            timeout=1800)
    parts = [p for p in glob.glob(os.path.join(out_dir, f"{name}_mesh*")) if not p.endswith(".zip")]
    out = os.path.join(out_dir, f"{name}_mesh.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in parts:
            z.write(p, os.path.basename(p))
    for p in parts:
        os.remove(p)
    return out


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """binary little-endian PLY vertices -> (xyz float32 Nx3, rgb uint8 Nx3 or None)"""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt, count, props, in_vertex = "", 0, [], False
        while True:
            words = f.readline().decode("ascii", "replace").split()
            if not words:
                continue
            if words[0] == "end_header":
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                in_vertex = words[1] == "vertex"
                if in_vertex:
                    count = int(words[2])
            elif words[0] == "property" and in_vertex:
                if words[1] == "list":
                    raise ValueError("list properties in vertices are not supported")
                props.append((words[2], _PLY_TYPES[words[1]]))
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt}")
        v = np.fromfile(f, dtype=np.dtype(props), count=count)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    names = v.dtype.names or ()
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8) if "red" in names else None
    return xyz, rgb


def voxel_down(xyz: np.ndarray, rgb: np.ndarray | None, size: float) -> tuple[np.ndarray, np.ndarray | None]:
    """one point per occupied voxel: the voxel centre, coloured with the mean colour of its points"""
    if len(xyz) == 0:
        return xyz, rgb
    q = np.floor(xyz / size).astype(np.int64)
    q -= q.min(0)
    span = q.max(0) + 1
    key = (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    centres = ((np.floor(xyz[first] / size) + 0.5) * size).astype(np.float32)
    if rgb is None:
        return centres, None
    n = np.bincount(inv, minlength=len(uniq)).astype(np.float32)
    cols = np.stack([np.bincount(inv, weights=rgb[:, c], minlength=len(uniq)) / n for c in range(3)], 1)
    return centres, cols.astype(np.uint8)


def list_exports() -> list[dict]:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    out = []
    for p in sorted(glob.glob(os.path.join(EXPORT_DIR, "*")), key=os.path.getmtime, reverse=True):
        if p.endswith((".ply", ".zip")):
            st = os.stat(p)
            out.append({"name": os.path.basename(p), "size_mb": round(st.st_size / 1e6, 1),
                        "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
    return out
