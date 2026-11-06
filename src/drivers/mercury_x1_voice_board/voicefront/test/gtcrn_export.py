"""Writes model/gtcrn.bin (the weights of the hand-written GTCRN in src/gtcrn_native.cpp) from the streaming ONNX
model gtcrn_simple.onnx (github.com/Xiaobin-Rong/gtcrn, stream/onnx_models). Also a numpy reference of the graph:
it is run against onnxruntime first, so a mismatch in the reading of the graph shows up before anything is written.

usage: gtcrn_export.py model/gtcrn.onnx model/gtcrn.bin
needs: numpy, onnx, onnxruntime (uv venv + uv pip install onnx onnxruntime numpy)

Blob format (little endian): magic "GTCRNW01", u32 n_records, then per record:
  u32 name_len, name bytes, u32 ndim, u32 dims[ndim], f32 data[prod(dims)]
All weights are already preprocessed for the C++ kernels (BN folded, ConvTranspose depthwise flipped into plain
conv form, MatMul weights stored output-major so every output is one dot product).
"""
import struct, sys, wave
import numpy as np
import onnx
from onnx import numpy_helper

def load_graph(path):
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i).astype(np.float32) if i.data_type == 1 else numpy_helper.to_array(i)
            for i in m.graph.initializer}
    by_op = {}
    for n in m.graph.node:
        by_op.setdefault(n.op_type, []).append(n)
    return init, by_op

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

class Weights:
    """Collects weights in the order the C++ reads them; also runs the reference forward."""
    def __init__(self, init, by_op):
        self.rec = []  # (name, array)
        I = init
        conv = by_op["Conv"]; convt = by_op["ConvTranspose"]; gru = by_op["GRU"]; mm = by_op["MatMul"]
        prelu = by_op["PRelu"]; bn = by_op["BatchNormalization"]
        adds = [n for n in by_op["Add"] if any(x in I and I[x].ndim == 1 for x in n.input)]
        def addbias(node):  # the initializer input of an Add node
            return [I[x] for x in node.input if x in I][0]
        def bnfold(node):  # returns (scale, shift) per channel
            w, b, mean, var = (I[x] for x in node.input[1:5])
            eps = [a.f for a in node.attribute if a.name == "epsilon"][0]
            s = w / np.sqrt(var + eps)
            return s, b - mean * s
        def gru_w(node, hidden):
            W, R, B = I[node.input[1]], I[node.input[2]], I[node.input[3]]
            return W, R, B  # [D,3H,In] [D,3H,H] [D,6H]
        # --- ERB
        self.erb = I[mm[0].input[1]].T.copy()      # [64,192]: erb[e] . x_high
        self.ierb = I[mm[-1].input[1]].T.copy()    # [192,64]: ierb[f] . m_high
        # --- encoder conv1 (9->16, k5 s2) + conv2 (16->16 groups 2, k5 s2)
        self.c1w, self.c1b, self.c1a = I[conv[0].input[1]][:, :, 0, :], I[conv[0].input[2]], float(I[prelu[0].input[1]].ravel()[0])
        self.c2w, self.c2b, self.c2a = I[conv[1].input[1]][:, :, 0, :], I[conv[1].input[2]], float(I[prelu[1].input[1]].ravel()[0])
        # --- encoder GT blocks
        self.enc = []
        for i in range(3):
            c = conv[2 + 3 * i: 5 + 3 * i]
            p = prelu[2 + 2 * i: 4 + 2 * i]
            blk = dict(
                pc1w=c[0].input[1] and I[c[0].input[1]][:, :, 0, 0], pc1b=I[c[0].input[2]], pc1a=float(I[p[0].input[1]].ravel()[0]),
                dww=I[c[1].input[1]][:, 0], dwb=I[c[1].input[2]], dwa=float(I[p[1].input[1]].ravel()[0]),
                pc2w=I[c[2].input[1]][:, :, 0, 0], pc2b=I[c[2].input[2]])
            W, R, B = gru_w(gru[i], 16)
            blk.update(gW=W[0], gR=R[0], gB=B[0])
            blk.update(fcw=I[mm[1 + i].input[1]].T.copy(), fcb=addbias(adds[i]))
            self.enc.append(blk)
        # --- DPGRNN x2
        self.dp = []
        g_i = 3; mm_i = 4; add_i = 3
        for s in range(2):
            d = {}
            for k, name in enumerate(["intra_g1", "intra_g2"]):
                W, R, B = gru_w(gru[g_i], 4); g_i += 1
                d[name] = (W, R, B)  # [2,12,8] [2,12,4] [2,24]
            d["intra_fcw"] = I[mm[mm_i].input[1]].T.copy(); d["intra_fcb"] = addbias(adds[add_i]); mm_i += 1; add_i += 1
            d["intra_lnw"] = I[f"dpgrnn{s+1}.intra_ln.weight"]; d["intra_lnb"] = I[f"dpgrnn{s+1}.intra_ln.bias"]
            for k, name in enumerate(["inter_g1", "inter_g2"]):
                W, R, B = gru_w(gru[g_i], 8); g_i += 1
                d[name] = (W[0], R[0], B[0])  # [24,8] [24,8] [48]
            d["inter_fcw"] = I[mm[mm_i].input[1]].T.copy(); d["inter_fcb"] = addbias(adds[add_i]); mm_i += 1; add_i += 1
            d["inter_lnw"] = I[f"dpgrnn{s+1}.inter_ln.weight"]; d["inter_lnb"] = I[f"dpgrnn{s+1}.inter_ln.bias"]
            self.dp.append(d)
        # --- decoder GT blocks (ConvTranspose + BN, folded)
        self.dec = []
        for i in range(3):
            c = convt[3 * i: 3 * i + 3]; b = bn[3 * i: 3 * i + 3]; p = prelu[8 + 2 * i: 10 + 2 * i]
            s1, t1 = bnfold(b[0]); s2, t2 = bnfold(b[1]); s3, t3 = bnfold(b[2])
            w1 = I[c[0].input[1]][:, :, 0, 0].T * s1[:, None]            # [16 out, 24 in]
            b1 = I[c[0].input[2]] * s1 + t1
            dw = I[c[1].input[1]][:, 0]                                   # [16,3,3] convT weights
            dw = dw[:, ::-1, ::-1].copy() * s2[:, None, None]             # flipped -> plain causal conv
            b2 = I[c[1].input[2]] * s2 + t2
            w3 = I[c[2].input[1]][:, :, 0, 0].T * s3[:, None]            # [8 out, 16 in]
            b3 = I[c[2].input[2]] * s3 + t3
            blk = dict(pc1w=w1, pc1b=b1, pc1a=float(I[p[0].input[1]].ravel()[0]), dww=dw, dwb=b2,
                       dwa=float(I[p[1].input[1]].ravel()[0]), pc2w=w3, pc2b=b3)
            W, R, B = gru_w(gru[11 + i], 16)
            blk.update(gW=W[0], gR=R[0], gB=B[0])
            blk.update(fcw=I[mm[8 + i].input[1]].T.copy(), fcb=addbias(adds[7 + i]))
            self.dec.append(blk)
        # --- decoder tail: convT 16->16 groups 2 k5 s2 (33->65) + BN + PReLU ; convT 16->2 k5 s2 (65->129) + BN + tanh
        s, t = bnfold(bn[9])
        self.d3w = I[convt[9].input[1]][:, :, 0, :].copy()               # [16 in, 8 out/g, 5]
        self.d3w = self.d3w * s.reshape(2, 8)[None, :, None].repeat(8, 0).reshape(16, 8, 1)  # scale by out channel
        # careful: out channel o = g*8 + j for in channel i in group g = i // 8
        self.d3w = I[convt[9].input[1]][:, :, 0, :].copy()
        for i in range(16):
            g = i // 8
            self.d3w[i] *= s[g * 8:(g + 1) * 8, None]
        self.d3b = I[convt[9].input[2]] * s + t
        self.d3a = float(I[prelu[14].input[1]].ravel()[0])
        s, t = bnfold(bn[10])
        self.d4w = I[convt[10].input[1]][:, :, 0, :] * s[None, :, None]  # [16 in, 2 out, 5]
        self.d4b = I[convt[10].input[2]] * s + t
        # sanity on the SFE gather index tables
        for n in by_op["Gather"]:
            if n.input[1] in I and I[n.input[1]].ndim == 2 and I[n.input[1]].shape[0] == 3:
                idx = I[n.input[1]]
                F = idx.shape[1]
                assert np.array_equal(idx, np.arange(3)[:, None] + np.arange(F)[None, :]), "unexpected SFE index"

    # ---------------- forward ----------------
    @staticmethod
    def prelu(x, a): return np.where(x >= 0, x, a * x)

    @staticmethod
    def gru_cell(x, h, W, R, B):
        """ONNX GRU (linear_before_reset=1), gate order z,r,h. x [..,In], h [..,H]."""
        H = h.shape[-1]
        gx = x @ W.T + B[:3 * H]
        gh = h @ R.T + B[3 * H:]
        z = sigmoid(gx[..., :H] + gh[..., :H])
        r = sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H])
        n = np.tanh(gx[..., 2 * H:] + r * gh[..., 2 * H:])
        return (1 - z) * n + z * h

    @staticmethod
    def sfe(x):  # x [C,F] -> [3C,F], channel c*3+k = x[c, f+k-1]
        C, F = x.shape
        xp = np.pad(x, ((0, 0), (1, 1)))
        return np.stack([xp[:, k:k + F] for k in range(3)], 1).reshape(3 * C, F)

    def gt_block(self, blk, x, ccache, tcache, dil):
        """x [16,33]; ccache [16, 2*dil, 33] (oldest first); tcache [16]. Returns y, new caches."""
        x1, x2 = x[:8], x[8:]
        h = self.prelu(blk["pc1w"] @ self.sfe(x1) + blk["pc1b"][:, None], blk["pc1a"])  # [16,33]
        frames = np.concatenate([ccache, h[:, None, :]], 1)  # [16, 2d+1, 33]
        new_cc = frames[:, 1:].copy()
        fp = np.pad(frames, ((0, 0), (0, 0), (1, 1)))
        y = np.zeros((16, 33), np.float32)
        for kt in range(3):
            for kf in range(3):
                y += blk["dww"][:, kt, kf][:, None] * fp[:, kt * dil, kf:kf + 33]
        y = self.prelu(y + blk["dwb"][:, None], blk["dwa"])
        h2 = blk["pc2w"] @ y + blk["pc2b"][:, None]  # [8,33]
        zt = (h2 ** 2).mean(1)  # [8]
        hn = self.gru_cell(zt, tcache, blk["gW"], blk["gR"], blk["gB"])
        at = sigmoid(blk["fcw"] @ hn + blk["fcb"])  # [8]
        h1 = h2 * at[:, None]
        out = np.empty((16, 33), np.float32)
        out[0::2] = h1; out[1::2] = x2
        return out, new_cc, hn

    def dpgrnn(self, d, x, icache):
        """x [16,33] (C,F); icache [33,16]. Returns y [16,33], new cache."""
        xt = x.T  # [33,16] (F,C)
        outs = []
        for gi, name in enumerate(["intra_g1", "intra_g2"]):
            W, R, B = d[name]
            xin = xt[:, gi * 8:(gi + 1) * 8]
            fw = np.zeros((33, 4), np.float32); h = np.zeros(4, np.float32)
            for f in range(33):
                h = self.gru_cell(xin[f], h, W[0], R[0], B[0]); fw[f] = h
            bw = np.zeros((33, 4), np.float32); h = np.zeros(4, np.float32)
            for f in range(32, -1, -1):
                h = self.gru_cell(xin[f], h, W[1], R[1], B[1]); bw[f] = h
            outs.append(np.concatenate([fw, bw], 1))
        intra = np.concatenate(outs, 1) @ d["intra_fcw"].T + d["intra_fcb"]  # [33,16]
        intra = self.ln(intra, d["intra_lnw"], d["intra_lnb"])
        x1 = xt + intra
        outs = []; newc = np.empty((33, 16), np.float32)
        for gi, name in enumerate(["inter_g1", "inter_g2"]):
            W, R, B = d[name]
            h = self.gru_cell(x1[:, gi * 8:(gi + 1) * 8], icache[:, gi * 8:(gi + 1) * 8], W, R, B)  # [33,8]
            outs.append(h); newc[:, gi * 8:(gi + 1) * 8] = h
        inter = np.concatenate(outs, 1) @ d["inter_fcw"].T + d["inter_fcb"]
        inter = self.ln(inter, d["inter_lnw"], d["inter_lnb"])
        return (x1 + inter).T, newc

    @staticmethod
    def ln(x, w, b):
        mu = x.mean(); v = ((x - mu) ** 2).mean()
        return (x - mu) / np.sqrt(v + 1e-8) * w + b

    def forward(self, mix, conv_cache, tra_cache, inter_cache):
        """mix [257,2]; caches as in the ONNX model (copied). Returns enh [257,2] and new caches."""
        conv_cache = conv_cache.copy(); tra_cache = tra_cache.copy(); inter_cache = inter_cache.copy()
        re, im = mix[:, 0], mix[:, 1]
        mag = np.sqrt(re ** 2 + im ** 2 + 1e-12)
        feat = np.stack([mag, re, im])  # [3,257]
        feat = np.concatenate([feat[:, :65], feat[:, 65:] @ self.erb.T], 1)  # [3,129]
        feat = self.sfe(feat)  # [9,129]
        # conv1: k5 s2 p2 -> 65
        fp = np.pad(feat, ((0, 0), (2, 2)))
        e0 = np.zeros((16, 65), np.float32)
        for k in range(5):
            e0 += self.c1w[:, :, k] @ fp[:, k:k + 2 * 65:2]
        e0 = self.prelu(e0 + self.c1b[:, None], self.c1a)
        # conv2: groups 2
        fp = np.pad(e0, ((0, 0), (2, 2)))
        e1 = np.zeros((16, 33), np.float32)
        for g in range(2):
            for k in range(5):
                e1[g * 8:(g + 1) * 8] += self.c2w[g * 8:(g + 1) * 8, :, k] @ fp[g * 8:(g + 1) * 8, k:k + 2 * 33:2]
        e1 = self.prelu(e1 + self.c2b[:, None], self.c2a)
        en = [e0, e1]
        x = e1
        cc = conv_cache[0, 0]  # [16,16,33]
        slices = [(0, 2, 1), (2, 6, 2), (6, 16, 5)]
        for i, (a, b, dil) in enumerate(slices):
            x, cc[:, a:b], tra_cache[0, i, 0, 0] = self.gt_block(self.enc[i], x, cc[:, a:b], tra_cache[0, i, 0, 0], dil)
            en.append(x)
        x, inter_cache[0, 0] = self.dpgrnn(self.dp[0], x, inter_cache[0, 0])
        x, inter_cache[1, 0] = self.dpgrnn(self.dp[1], x, inter_cache[1, 0])
        cc = conv_cache[1, 0]
        for i, (a, b, dil) in enumerate(reversed(slices)):
            x, cc[:, a:b], tra_cache[1, i, 0, 0] = self.gt_block(self.dec[i], x + en[4 - i], cc[:, a:b], tra_cache[1, i, 0, 0], dil)
        # de_convs[3]: convT groups 2, k5 s2 p2: 33 -> 65 ; out[f] = sum_{fi,k: 2fi-2+k=f} W[i,j,k] x[i,fi]
        x = x + en[1]
        y = np.zeros((16, 65), np.float32)
        for i in range(16):
            g = i // 8
            for fi in range(33):
                for k in range(5):
                    fo = 2 * fi - 2 + k
                    if 0 <= fo < 65:
                        y[g * 8:(g + 1) * 8, fo] += self.d3w[i, :, k] * x[i, fi]
        y = self.prelu(y + self.d3b[:, None], self.d3a) + en[0]
        m = np.zeros((2, 129), np.float32)
        for i in range(16):
            for fi in range(65):
                for k in range(5):
                    fo = 2 * fi - 2 + k
                    if 0 <= fo < 129:
                        m[:, fo] += self.d4w[i, :, k] * y[i, fi]
        m = np.tanh(m + self.d4b[:, None])
        m = np.concatenate([m[:, :65], m[:, 65:] @ self.ierb.T], 1)  # [2,257]
        enh = np.stack([re * m[0] - im * m[1], im * m[0] + re * m[1]], 1)
        return enh.astype(np.float32), conv_cache, tra_cache, inter_cache

    # ---------------- blob ----------------
    def records(self):
        r = [("erb", self.erb), ("ierb", self.ierb), ("c1w", self.c1w), ("c1b", self.c1b), ("c1a", np.array([self.c1a])),
             ("c2w", self.c2w), ("c2b", self.c2b), ("c2a", np.array([self.c2a]))]
        for tag, blocks in (("enc", self.enc), ("dec", self.dec)):
            for i, b in enumerate(blocks):
                for k in ["pc1w", "pc1b", "pc1a", "dww", "dwb", "dwa", "pc2w", "pc2b", "gW", "gR", "gB", "fcw", "fcb"]:
                    v = b[k]
                    r.append((f"{tag}{i}.{k}", np.array([v]) if isinstance(v, float) else v))
        for s, d in enumerate(self.dp):
            for name in ["intra_g1", "intra_g2", "inter_g1", "inter_g2"]:
                W, R, B = d[name]
                r += [(f"dp{s}.{name}.W", W), (f"dp{s}.{name}.R", R), (f"dp{s}.{name}.B", B)]
            for k in ["intra_fcw", "intra_fcb", "intra_lnw", "intra_lnb", "inter_fcw", "inter_fcb", "inter_lnw", "inter_lnb"]:
                r.append((f"dp{s}.{k}", d[k]))
        r += [("d3w", self.d3w), ("d3b", self.d3b), ("d3a", np.array([self.d3a])), ("d4w", self.d4w), ("d4b", self.d4b)]
        return r

    def write_blob(self, path):
        recs = self.records()
        with open(path, "wb") as f:
            f.write(b"GTCRNW01"); f.write(struct.pack("<I", len(recs)))
            for name, a in recs:
                a = np.ascontiguousarray(a, dtype=np.float32)
                nb = name.encode()
                f.write(struct.pack("<I", len(nb))); f.write(nb)
                f.write(struct.pack("<I", a.ndim)); f.write(struct.pack(f"<{a.ndim}I", *a.shape))
                f.write(a.tobytes())
        print(f"wrote {path}: {len(recs)} records, {sum(a.size for _, a in recs)} floats")


def main():
    import onnxruntime as ort
    path, out = sys.argv[1], sys.argv[2]
    init, by_op = load_graph(path)
    w = Weights(init, by_op)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    caches = [np.zeros((2, 1, 16, 16, 33), np.float32), np.zeros((2, 3, 1, 1, 16), np.float32), np.zeros((2, 1, 33, 16), np.float32)]
    mycaches = [c.copy() for c in caches]
    worst = 0
    for f in range(40):
        mix = (rng.standard_normal((257, 2)) * 3).astype(np.float32)
        enh, *c = s.run(None, {"mix": mix.reshape(1, 257, 1, 2), "conv_cache": caches[0], "tra_cache": caches[1], "inter_cache": caches[2]})
        caches = c
        my, *mc = w.forward(mix, *mycaches)
        mycaches = mc
        err = np.abs(my - enh[0, :, 0, :]).max() / (np.abs(enh).max() + 1e-9)
        cerr = max(np.abs(a - b).max() for a, b in zip(mycaches, caches))
        worst = max(worst, err, cerr)
    print(f"random frames: max rel err out {err:.2e}, caches {cerr:.2e}, worst {worst:.2e}")
    w.write_blob(out)


if __name__ == "__main__":
    main()
