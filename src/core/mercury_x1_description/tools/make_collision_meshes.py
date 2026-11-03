#!/usr/bin/env python3
"""Collision meshes for MoveIt: one convex hull per link, a few hundred triangles (run off-robot; outputs committed).

The vendor URDF collides with the full visual COLLADA meshes (~125 MB, 100k+ triangles per link): far too slow for
collision checking on the Jetson. Each hull is built from the link's visual mesh with the visual origin baked in
(so it sits in the link frame), padded by PAD outward, and the URDF's <collision> blocks are rewritten to use it.

  uv run --no-project --with trimesh --with pycollada --with scipy --with lxml --with networkx --with fast-simplification \
      tools/make_collision_meshes.py urdf/mercury_x1_vendor.urdf ../mercury_robot_urdf/mercury_robot_urdf meshes/collision
"""
from __future__ import annotations

import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import trimesh.transformations as tf

MAX_FACES = 300     # per hull
PAD = 0.005         # m outward: covers decimation shrink + cables


def origin_matrix(el) -> np.ndarray:
    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (el.get("rpy") or "0 0 0").split()]
    m = tf.euler_matrix(*rpy, axes="sxyz")
    m[:3, 3] = xyz
    return m


def hull(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    h = mesh.convex_hull
    if len(h.faces) > MAX_FACES:
        h = h.simplify_quadric_decimation(face_count=MAX_FACES).convex_hull
    # pad: push every vertex out from the centroid by PAD (hull stays convex)
    c = h.centroid
    d = h.vertices - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return trimesh.Trimesh(c + d * (1 + PAD / np.maximum(n, 1e-6)), h.faces).convex_hull


def main(urdf: str, pkg_root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    text = open(urdf).read()
    root = ET.fromstring(text)
    for link in root.findall("link"):
        name = link.get("name")
        vis = link.find("visual")
        if vis is None or vis.find("geometry/mesh") is None:
            continue
        mesh_el = vis.find("geometry/mesh")
        path = re.sub(r"^package://mercury_robot_urdf/", "", mesh_el.get("filename"))
        fn = os.path.join(pkg_root, path)
        scene = trimesh.load(fn, force="scene")
        mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
        # trimesh ignores COLLADA <unit>; the vendor files are in millimetres
        unit = re.search(r'<unit[^>]*meter="([0-9.eE+-]+)"', open(fn, errors="ignore").read(4096))
        mesh.apply_scale(float(unit.group(1)) if unit else 1.0)
        if mesh_el.get("scale"):
            mesh.apply_scale([float(v) for v in mesh_el.get("scale").split()])
        mesh.apply_transform(origin_matrix(vis.find("origin")))
        h = hull(mesh)
        h.export(os.path.join(out_dir, f"{name}.stl"))
        ext = h.bounds[1] - h.bounds[0]
        print(f"{name:12s} {len(mesh.faces):7d} -> {len(h.faces):4d} faces, extent {np.round(ext, 3)} m")
        # <collision> of this link -> the hull, in the link frame
        new = (f'<collision>\n      <geometry>\n        <mesh filename="package://mercury_x1_description/meshes/collision/'
               f'{name}.stl"/>\n      </geometry>\n      <origin xyz="0 0 0" rpy="0 0 0"/>\n    </collision>')
        pat = re.compile(r'(<link name\s*=\s*"' + re.escape(name) + r'"\s*>.*?)<collision>.*?</collision>', re.S)
        text, n = pat.subn(lambda m: m.group(1) + new, text, count=1)
        if not n:
            print(f"  (no <collision> block for {name} in the URDF)")
    open(urdf, "w").write(text)


if __name__ == "__main__":
    main(*sys.argv[1:4])
