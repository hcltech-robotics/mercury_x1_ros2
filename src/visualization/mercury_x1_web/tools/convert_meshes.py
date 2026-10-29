#!/usr/bin/env python3
"""Convert the Mercury X1 URDF visuals (COLLADA) into per-link GLB files for Rerun.

Each GLB is expressed in its link frame (visual origin baked in) and simplified to keep
the robot model light enough to stream to a browser.

  python convert_meshes.py <urdf> <mesh_dir_for_package://mercury_robot_urdf> <out_dir> [--faces 15000]
Needs: trimesh pycollada fast-simplification lxml networkx (run it off-robot, outputs are committed).
"""
import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import trimesh.transformations as tf


# The vendor COLLADA files carry no usable materials; give the model a readable palette.
PALETTE = {"base_link": [70, 74, 82], "link_body": [214, 217, 222], "link_head": [236, 238, 241],
           "link_eye": [40, 42, 46]}
ARM = [[200, 204, 210], [120, 126, 134]]


def link_color(name, fallback):
    if name in PALETTE:
        return PALETTE[name]
    if name.startswith("link") and name[4:5].isdigit():
        return ARM[int(name[4]) % 2]
    return fallback or [200, 200, 200]


def rpy_matrix(rpy):
    r, p, y = rpy
    return tf.euler_matrix(r, p, y, "sxyz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("package_root", help="directory that package://mercury_robot_urdf maps to")
    ap.add_argument("out")
    ap.add_argument("--faces", type=int, default=40000, help="max faces per link")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = ET.parse(a.urdf).getroot()
    index = {}
    for link in root.findall("link"):
        name = link.get("name")
        vis = link.find("visual")
        if vis is None:
            continue
        mesh_el = vis.find("geometry/mesh")
        if mesh_el is None:
            continue
        fn = mesh_el.get("filename").replace("package://mercury_robot_urdf", a.package_root)
        org = vis.find("origin")
        xyz = [float(v) for v in (org.get("xyz", "0 0 0").split() if org is not None else [0, 0, 0])]
        rpy = [float(v) for v in (org.get("rpy", "0 0 0").split() if org is not None else [0, 0, 0])]
        scene = trimesh.load(fn, force="scene")
        mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
        color = None
        try:
            vc = mesh.visual.to_color().vertex_colors
            color = np.median(vc[:, :3], axis=0).astype(int).tolist()
        except Exception:
            pass
        # trimesh ignores COLLADA <unit>; the vendor files are in millimetres
        m = re.search(r'<unit[^>]*meter="([0-9.eE+-]+)"', open(fn, errors="ignore").read(4096))
        mesh.apply_scale(float(m.group(1)) if m else 1.0)
        # COLLADA exports unshared vertices per triangle; merge them or decimation tears the surface
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
        n0 = len(mesh.faces)
        if n0 > a.faces:
            mesh = mesh.simplify_quadric_decimation(face_count=a.faces)
        T = rpy_matrix(rpy)
        T[:3, 3] = xyz
        mesh.apply_transform(T)
        color = link_color(name, color)
        rgba = np.array(color + [255], dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)))
        out = os.path.join(a.out, f"{name}.glb")
        mesh.export(out, include_normals=True)
        index[name] = {"file": f"{name}.glb", "faces": len(mesh.faces), "faces_orig": n0, "color": color}
        print(f"{name:12s} {n0:7d} -> {len(mesh.faces):6d} faces  {os.path.getsize(out) / 1e6:5.2f} MB")
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
