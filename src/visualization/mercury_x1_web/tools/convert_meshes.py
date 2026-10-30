#!/usr/bin/env python3
"""Convert the Mercury X1 URDF visuals (COLLADA) into per-link GLB files for Rerun.

Each GLB is expressed in its link frame (visual origin baked in) and simplified to keep
the robot model light enough to stream to a browser.

  python convert_meshes.py <urdf> <mesh_dir_for_package://mercury_robot_urdf> <out_dir> [--faces 15000]
  python convert_meshes.py --recolor <out_dir> [--color 222,225,230]   # repaint existing GLBs only
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


# The vendor COLLADA files carry no usable materials: paint the whole robot one body colour
# (per-link palettes made the base, torso and arm links look like separate objects).
BODY_COLOR = [222, 225, 230]


def recolor(out_dir: str, color: list[int]) -> None:
    """Repaint the GLBs in out_dir (geometry and normals untouched) and update index.json."""
    rgba = np.array(color + [255], dtype=np.uint8)
    with open(os.path.join(out_dir, "index.json")) as f:
        index = json.load(f)
    for name, info in index.items():
        for fname in info.get("files") or [info["file"]]:
            path = os.path.join(out_dir, fname)
            mesh = trimesh.load(path, force="mesh", process=False)
            normals = np.asarray(mesh.vertex_normals)
            part = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_normals=normals, process=False)
            part.visual = trimesh.visual.ColorVisuals(part, vertex_colors=np.tile(rgba, (len(part.vertices), 1)))
            part.export(path, include_normals=True)
        info["color"] = color
        print(f"{name:12s} -> {color}")
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


def rpy_matrix(rpy):
    r, p, y = rpy
    return tf.euler_matrix(r, p, y, "sxyz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recolor", metavar="OUT_DIR", help="only repaint the GLBs already in OUT_DIR")
    ap.add_argument("--color", default=",".join(map(str, BODY_COLOR)), help="body colour r,g,b")
    ap.add_argument("urdf", nargs="?")
    ap.add_argument("package_root", nargs="?", help="directory that package://mercury_robot_urdf maps to")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--faces", type=int, default=150000, help="max faces per link")
    ap.add_argument("--crease", type=float, default=30.0, help="normals are smooth across edges flatter than this [deg]")
    ap.add_argument("--chunk", type=int, default=40000,
                    help="max faces per GLB: Rerun 0.32's gRPC server blocks forever on single messages of a few MB")
    a = ap.parse_args()
    body = [int(v) for v in a.color.split(",")]
    if a.recolor:
        recolor(a.recolor, body)
        return
    if not (a.urdf and a.package_root and a.out):
        ap.error("urdf, package_root and out are required unless --recolor is given")
    os.makedirs(a.out, exist_ok=True)
    for old in os.listdir(a.out):
        if old.endswith(".glb"):
            os.remove(os.path.join(a.out, old))
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
        # smooth shading within surfaces, sharp at real CAD edges
        mesh = trimesh.graph.smooth_shade(mesh, angle=math.radians(a.crease), facet_minarea=None)
        T = rpy_matrix(rpy)
        T[:3, 3] = xyz
        mesh.apply_transform(T)
        color = body
        rgba = np.array(color + [255], dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)))
        normals = np.asarray(mesh.vertex_normals)
        files, size = [], 0
        for k, start in enumerate(range(0, len(mesh.faces), a.chunk)):
            faces = mesh.faces[start:start + a.chunk]
            used, inv = np.unique(faces.ravel(), return_inverse=True)
            part = trimesh.Trimesh(vertices=mesh.vertices[used], faces=inv.reshape(-1, 3),
                                   vertex_normals=normals[used], process=False)
            part.visual = trimesh.visual.ColorVisuals(part, vertex_colors=np.tile(rgba, (len(used), 1)))
            fname = f"{name}.glb" if len(mesh.faces) <= a.chunk else f"{name}_{k}.glb"
            part.export(os.path.join(a.out, fname), include_normals=True)
            files.append(fname)
            size += os.path.getsize(os.path.join(a.out, fname))
        index[name] = {"files": files, "faces": len(mesh.faces), "faces_orig": n0, "color": color}
        print(f"{name:12s} {n0:7d} -> {len(mesh.faces):6d} faces  {len(files)} part(s)  {size / 1e6:5.2f} MB")
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
