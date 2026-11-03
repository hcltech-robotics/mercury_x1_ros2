#!/usr/bin/env python3
"""Self-collision matrix for the SRDF: which link pairs never need checking (run off-robot, result committed).

Same rules as the MoveIt Setup Assistant (whose collisions_updater took > 15 min on one core here):
  Adjacent  parent/child links (across fixed joints and geometry-less links)
  Always    in collision in >= 95 % of random poses
  Never     never in collision in N random poses (joints uniform within the URDF limits)
Forward kinematics in numpy, collisions with FCL on the convex-hull collision meshes, spread over all cores.
Rewrites the <disable_collisions> block of the SRDF.

  xacro src/core/mercury_x1_description/urdf/mercury_x1.urdf.xacro > /tmp/mx1.urdf
  uv run --no-project --with trimesh --with python-fcl --with numpy --with scipy \
      src/control/mercury_x1_moveit_config/scripts/make_acm.py /tmp/mx1.urdf src/core/mercury_x1_description \
      src/control/mercury_x1_moveit_config/config/mercury_x1.srdf [samples]
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import trimesh.transformations as tf

ALWAYS = 0.95


def origin(el) -> np.ndarray:
    if el is None:
        return np.eye(4)
    m = tf.euler_matrix(*[float(v) for v in (el.get("rpy") or "0 0 0").split()], axes="sxyz")
    m[:3, 3] = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
    return m


class Robot:
    def __init__(self, urdf: str, desc_dir: str) -> None:
        r = ET.parse(urdf).getroot()
        self.joints = []          # (name, type, parent, child, origin, axis, lo, hi) in tree order
        for j in r.findall("joint"):
            lim = j.find("limit")
            ax = j.find("axis")
            self.joints.append((j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                                origin(j.find("origin")),
                                np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
                                float(lim.get("lower")) if lim is not None else 0.0,
                                float(lim.get("upper")) if lim is not None else 0.0))
        children = {j[3] for j in self.joints}
        self.root = next(j[2] for j in self.joints if j[2] not in children)
        self.meshes = {}
        for link in r.findall("link"):
            m = link.find("collision/geometry/mesh")
            if m is not None:
                fn = re.sub(r"^package://mercury_x1_description/", "", m.get("filename"))
                self.meshes[link.get("name")] = trimesh.load(os.path.join(desc_dir, fn), force="mesh")
        self.moving = [j for j in self.joints if j[1] in ("revolute", "continuous", "prismatic")]

    def fk(self, q: dict) -> dict:
        T = {self.root: np.eye(4)}
        pending = list(self.joints)
        while pending:
            rest = []
            for name, typ, parent, child, o, axis, _, _ in pending:
                if parent not in T:
                    rest.append((name, typ, parent, child, o, axis, _, _))
                    continue
                m = T[parent] @ o
                if typ in ("revolute", "continuous"):
                    m = m @ tf.rotation_matrix(q.get(name, 0.0), axis)
                T[child] = m
            if len(rest) == len(pending):
                break
            pending = rest
        return T

    def adjacent(self) -> set:
        """parent/child pairs between links with geometry, looking through links without geometry"""
        parent = {j[3]: j[2] for j in self.joints}

        def geo_parent(link):
            p = parent.get(link)
            while p is not None and p not in self.meshes:
                p = parent.get(p)
            return p
        return {tuple(sorted((l, geo_parent(l)))) for l in self.meshes if geo_parent(l)}


_robot = None


def _init(urdf, desc):
    global _robot
    _robot = Robot(urdf, desc)


def _count(args) -> dict:
    seed, n = args
    rng = np.random.default_rng(seed)
    cm = trimesh.collision.CollisionManager()
    for name, mesh in _robot.meshes.items():
        cm.add_object(name, mesh)
    counts: dict = {}
    for _ in range(n):
        q = {j[0]: rng.uniform(j[6], j[7]) for j in _robot.moving}
        for name, T in _robot.fk(q).items():
            if name in _robot.meshes:
                cm.set_transform(name, T)
        _, pairs = cm.in_collision_internal(return_names=True)
        for p in pairs:
            k = tuple(sorted(p))
            counts[k] = counts.get(k, 0) + 1
    return counts


def main(urdf: str, desc: str, srdf: str, samples: int = 20000) -> None:
    robot = Robot(urdf, desc)
    links = sorted(robot.meshes)
    adjacent = robot.adjacent()
    workers = os.cpu_count() or 4
    per = samples // workers + 1
    with mp.Pool(workers, initializer=_init, initargs=(urdf, desc)) as pool:
        parts = pool.map(_count, [(s, per) for s in range(workers)])
    total = per * workers
    counts: dict = {}
    for c in parts:
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
    rows, checked = [], []
    for a, b in itertools.combinations(links, 2):
        k = (a, b)
        frac = counts.get(k, 0) / total
        if k in adjacent:
            rows.append((a, b, "Adjacent"))
        elif frac >= ALWAYS:
            rows.append((a, b, "Always"))
        elif frac == 0:
            rows.append((a, b, "Never"))
        else:
            checked.append((a, b, frac))
    print(f"{total} poses, {len(links)} links, {len(rows)} pairs disabled "
          f"({sum(r[2] == 'Adjacent' for r in rows)} adjacent, {sum(r[2] == 'Always' for r in rows)} always, "
          f"{sum(r[2] == 'Never' for r in rows)} never), {len(checked)} checked:")
    for a, b, f in sorted(checked, key=lambda x: -x[2]):
        print(f"  {a:10s} {b:10s} {100 * f:6.2f} %")
    block = "\n".join(f'  <disable_collisions link1="{a}" link2="{b}" reason="{why}"/>' for a, b, why in rows)
    text = open(srdf).read()
    text = re.sub(r"\n\s*<!-- self-collision matrix.*?<!-- end self-collision matrix -->", "", text, flags=re.S)
    text = text.replace("</robot>", f"\n  <!-- self-collision matrix: make_acm.py, {total} poses -->\n{block}\n"
                                     "  <!-- end self-collision matrix -->\n</robot>")
    open(srdf, "w").write(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 20000)
