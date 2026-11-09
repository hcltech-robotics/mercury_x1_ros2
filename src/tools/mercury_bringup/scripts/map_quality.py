"""map_quality.py DIR OUT.png - DIR holds rtabmap-export output: opt_poses.txt + opt.pgm/.yaml + opt_cloud.ply (--poses --map --cloud), odom_poses.txt (--opt 3 --poses) and in.db (a COPY of the map, never the live one).
map quality: odometry vs optimized path, link residuals, 3D wall points vs the 2D grid; overlay picture"""
import math, sqlite3, struct, sys, zlib
import numpy as np
from PIL import Image, ImageDraw
D = sys.argv[1]; out = sys.argv[2]
def poses(p):
    a = np.loadtxt(p, comments="#"); ids = a[:, 8].astype(int)
    yaw = np.arctan2(2 * (a[:, 7] * a[:, 6] + a[:, 4] * a[:, 5]), 1 - 2 * (a[:, 5] ** 2 + a[:, 6] ** 2))
    return {i: (x, y, t) for i, x, y, t in zip(ids, a[:, 1], a[:, 2], yaw)}, a[:, 0]
opt, ts = poses(f"{D}/opt_poses.txt"); odo, _ = poses(f"{D}/odom_poses.txt")
ids = sorted(set(opt) & set(odo))
# odometry -> optimized correction (after aligning the first pose they share)
corr = np.array([math.hypot(opt[i][0] - odo[i][0], opt[i][1] - odo[i][1]) for i in ids])
print(f"nodes {len(ids)}; odometry vs optimized: median {np.median(corr):.2f} m, max {corr.max():.2f} m at node {ids[int(corr.argmax())]}")
# links: residual of each link against the optimized graph
con = sqlite3.connect(f"{D}/in.db")
rows = con.execute("select from_id, to_id, type, transform from Link").fetchall()
def tf(blob):
    b = blob
    try: b = zlib.decompress(blob)
    except Exception: pass
    m = np.frombuffer(b, np.float32)[:12].reshape(3, 4)
    return m[0, 3], m[1, 3], math.atan2(m[1, 0], m[0, 0])
names = {0: "neighbor", 1: "global", 2: "local_space", 3: "local_time"}
bad = []
for f, t, ty, blob in rows:
    if f not in opt or t not in opt or ty not in names: continue
    dx, dy, dth = tf(blob)
    xf, yf, tf_ = opt[f]; xt, yt, tt = opt[t]
    c, s = math.cos(tf_), math.sin(tf_)
    rx, ry = c * (xt - xf) + s * (yt - yf), -s * (xt - xf) + c * (yt - yf)
    rth = (tt - tf_ + math.pi) % (2 * math.pi) - math.pi
    e = math.hypot(rx - dx, ry - dy); ea = abs((rth - dth + math.pi) % (2 * math.pi) - math.pi)
    bad.append((e, math.degrees(ea), f, t, names[ty]))
bad.sort(reverse=True)
by = {}
for e, a, f, t, n in bad: by.setdefault(n, []).append(e)
for n, v in by.items(): print(f"{n:12s} links {len(v):4d}: residual median {np.median(v)*100:.1f} cm, p95 {np.percentile(v,95)*100:.1f} cm, max {max(v)*100:.1f} cm")
print("worst links (residual cm, deg, from, to, type):")
for e, a, f, t, n in bad[:10]: print(f"   {e*100:6.1f} cm {a:5.1f} deg  {f}->{t} {n}")
# neighbor steps: odometry step length vs time -> jumps (slip / ICP failure) 
od = np.array([[odo[i][0], odo[i][1], odo[i][2]] for i in ids]); step = np.hypot(*np.diff(od[:, :2], axis=0).T)
dth = np.degrees(np.abs((np.diff(od[:, 2]) + np.pi) % (2 * np.pi) - np.pi))
print(f"odometry steps: median {np.median(step)*100:.0f} cm, max {step.max()*100:.0f} cm; turn per step max {dth.max():.0f} deg")
# 2D grid
yaml = dict(l.split(":", 1) for l in open(f"{D}/opt.yaml") if ":" in l)
res = float(yaml["resolution"]); org = [float(v) for v in yaml["origin"].strip(" []\n").split(",")[:2]]
img = np.array(Image.open(f"{D}/opt.pgm")); H, W = img.shape
occ = img < 50; free = img > 200
print(f"grid {W}x{H} at {res} m, occupied {occ.sum()} cells, free {free.sum()}")
# distance (in cells, up to 8) from each cell to the nearest occupied cell, by repeated dilation
dist = np.full(occ.shape, 99, np.int16); cur = occ.copy(); dist[cur] = 0
for k in range(1, 9):
    nb = cur.copy(); nb[1:] |= cur[:-1]; nb[:-1] |= cur[1:]; nb[:, 1:] |= cur[:, :-1]; nb[:, :-1] |= cur[:, 1:]
    dist[nb & (dist == 99)] = k; cur = nb
# 3D cloud (binary ply x y z float + rgb)
with open(f"{D}/opt_cloud.ply", "rb") as f:
    hdr = b""
    while not hdr.endswith(b"end_header\n"): hdr += f.readline()
    n = int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
    lines = hdr.split(b"\n"); k0 = next(i for i, l in enumerate(lines) if l.startswith(b"element vertex"))
    props = []
    for l in lines[k0 + 1:]:
        if l.startswith(b"element"): break
        if l.startswith(b"property"): props.append(l.split()[1:])
    dt = np.dtype([(p[1].decode(), {"float": "<f4", "uchar": "u1", "int": "<i4", "uint": "<u4", "float32": "<f4", "uint8": "u1"}[p[0].decode()]) for p in props])
    data = f.read(); print(f"ply: {n} vertices, {dt.itemsize} B each, {len(data)} B of data")
    pts = np.frombuffer(data, dt, min(n, len(data) // dt.itemsize))
xyz = np.stack([pts["x"], pts["y"], pts["z"]], 1)
wall = xyz[(xyz[:, 2] > 0.25) & (xyz[:, 2] < 1.3)]
cx = ((wall[:, 0] - org[0]) / res).astype(int); cy = (H - 1 - (wall[:, 1] - org[1]) / res).astype(int)
ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H); cx, cy = cx[ok], cy[ok]
dd = dist[cy, cx] * res
print(f"3D points 0.25-1.3 m high: {len(dd)}; within 5 cm of a 2D wall {np.mean(dd<=0.05)*100:.0f} %, within 10 cm {np.mean(dd<=0.10)*100:.0f} %, "
      f"within 20 cm {np.mean(dd<=0.20)*100:.0f} %, farther than 40 cm {np.mean(dd>0.40)*100:.0f} %")
seen = np.zeros(occ.shape, bool); seen[cy, cx] = True
sd = seen.copy()
for k in range(2):
    nb = sd.copy(); nb[1:] |= sd[:-1]; nb[:-1] |= sd[1:]; nb[:, 1:] |= sd[:, :-1]; nb[:, :-1] |= sd[:, 1:]; sd = nb
print(f"2D wall cells with 3D points within 10 cm: {np.mean(sd[occ])*100:.0f} %  (the rest: lidar-only things - low obstacles, glass, legs)")
# overlay
rgb = np.stack([img] * 3, -1).astype(np.uint8); rgb[img == 205] = (60, 60, 70); rgb[free] = (235, 235, 235); rgb[occ] = (0, 0, 0)
col = np.where(dd[:, None] <= 0.10, [0, 170, 0], np.where(dd[:, None] <= 0.30, [255, 160, 0], [230, 0, 0])).astype(np.uint8)
rgb[cy, cx] = col
im = Image.fromarray(rgb).resize((W * 3, H * 3), Image.NEAREST); d = ImageDraw.Draw(im)
def P(x, y): return (3 * (x - org[0]) / res, 3 * (H - 1 - (y - org[1]) / res))
d.line([P(odo[i][0], odo[i][1]) for i in ids], fill=(255, 0, 255), width=2)
d.line([P(opt[i][0], opt[i][1]) for i in ids], fill=(0, 90, 255), width=2)
for e, a, f, t, nme in bad[:10]:
    if e > 0.10: d.line([P(*opt[f][:2]), P(*opt[t][:2])], fill=(255, 0, 0), width=4)
for gx in range(math.ceil(org[0]), math.floor(org[0] + W * res) + 1):
    X, _ = P(gx, 0); d.text((X + 2, 2), str(gx), fill=(0, 0, 200))
for gy in range(math.ceil(org[1]), math.floor(org[1] + H * res) + 1):
    _, Y = P(0, gy); d.text((2, Y), str(gy), fill=(0, 0, 200))
im.save(out)
