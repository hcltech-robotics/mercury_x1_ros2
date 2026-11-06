#include "gtcrn_native.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

// Written for the voice board's Cortex-A53 (in-order, 64-bit NEON datapath): every hot loop keeps 8 independent
// vector accumulators in flight so load and FMA latencies overlap, and the GRU gates go through a vectorised exp
// instead of libm (tanhf is 200 ns there). GCC vector extensions, so the same code is SSE on x86 for the tests.

namespace {

constexpr int kBins = 257;
constexpr int kLow = 65;    // bins passed around the ERB mapping unchanged
constexpr int kHigh = 192;  // bins mapped onto 64 ERB bands
constexpr int kErb = 64;
constexpr int F0 = 129;  // feature bins after ERB
constexpr int F1 = 65;   // after conv1 (stride 2)
constexpr int F2 = 33;   // after conv2 (stride 2): the width of all GT blocks and the DPGRNNs
constexpr int C = 16;    // channels

typedef float v4 __attribute__((vector_size(16)));
typedef int32_t v4i __attribute__((vector_size(16)));

inline v4 ld(const float* p) {
  v4 r;
  std::memcpy(&r, p, 16);
  return r;
}
inline void sv(float* p, v4 v) { std::memcpy(p, &v, 16); }
inline v4 bc(float x) { return v4{x, x, x, x}; }
inline float hsum(v4 v) { return (v[0] + v[1]) + (v[2] + v[3]); }

// exp on 4 lanes (Cephes polynomial, |rel err| < 2e-7), argument clamped to +-87 so the exponent never wraps
inline v4 exp4(v4 x) {
  const v4 hi = bc(87.0f), lo = bc(-87.0f);
  x = x > hi ? hi : x;
  x = x < lo ? lo : x;
  // round x*log2(e) to the nearest integer through an int conversion (a float "magic number" rounding would
  // be folded away under -ffast-math)
  const v4 zf = x * bc(1.44269504088896341f);
  const v4i e = __builtin_convertvector(zf + (zf >= bc(0.0f) ? bc(0.5f) : bc(-0.5f)), v4i);
  const v4 z = __builtin_convertvector(e, v4);
  const v4 r = x - z * bc(0.693359375f) - z * bc(-2.12194440e-4f);
  v4 p = bc(1.9875691500E-4f);
  p = p * r + bc(1.3981999507E-3f);
  p = p * r + bc(8.3334519073E-3f);
  p = p * r + bc(4.1665795894E-2f);
  p = p * r + bc(1.6666665459E-1f);
  p = p * r + bc(5.0000001201E-1f);
  p = p * r * r + r + bc(1.0f);
  v4i pb;
  std::memcpy(&pb, &p, 16);
  pb += e << 23;
  std::memcpy(&p, &pb, 16);
  return p;
}
inline v4 sigmoid4(v4 x) {
  const v4 zero = bc(0.0f);
  const v4 e = exp4(x >= zero ? -x : x);  // exp(-|x|): never overflows
  const v4 d = bc(1.0f) / (bc(1.0f) + e);
  return x >= zero ? d : e * d;
}
inline v4 tanh4(v4 x) {
  const v4 zero = bc(0.0f);
  const v4 e = exp4(bc(-2.0f) * (x >= zero ? x : -x));
  const v4 t = (bc(1.0f) - e) / (bc(1.0f) + e);
  return x >= zero ? t : -t;
}
inline void sigmoid_n(float* y, const float* x, int n) {  // n multiple of 4
  for (int i = 0; i < n; i += 4) sv(y + i, sigmoid4(ld(x + i)));
}
inline float prelu(float x, float a) { return x >= 0.0f ? x : a * x; }
inline void prelu_n(float* x, float a, int n) {
  for (int f = 0; f < n; ++f) x[f] = prelu(x[f], a);
}

// out[f] = bias + sum_i w[i] * rows[i][f] for f < n. The one kernel behind every convolution here: a pointwise
// conv is rows = input channels; a conv with frequency taps / stride adds one row pointer per tap (shifted into a
// padded or de-interleaved copy of the input); the depthwise conv has 9 taps. n is 32k or 32k+1 in this net, so
// the vector part runs in blocks of 8 vectors and at most one scalar element is left.
inline void rows_mac(float* out, float bias, const float* const* rows, const float* w, int nin, int n) {
  int f = 0;
  for (; f + 32 <= n; f += 32) {
    v4 a0 = bc(bias), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (int i = 0; i < nin; ++i) {
      const float* r = rows[i] + f;
      const v4 wv = bc(w[i]);
      a0 += wv * ld(r);
      a1 += wv * ld(r + 4);
      a2 += wv * ld(r + 8);
      a3 += wv * ld(r + 12);
      a4 += wv * ld(r + 16);
      a5 += wv * ld(r + 20);
      a6 += wv * ld(r + 24);
      a7 += wv * ld(r + 28);
    }
    sv(out + f, a0);
    sv(out + f + 4, a1);
    sv(out + f + 8, a2);
    sv(out + f + 12, a3);
    sv(out + f + 16, a4);
    sv(out + f + 20, a5);
    sv(out + f + 24, a6);
    sv(out + f + 28, a7);
  }
  for (; f + 4 <= n; f += 4) {
    v4 a = bc(bias);
    for (int i = 0; i < nin; ++i) a += bc(w[i]) * ld(rows[i] + f);
    sv(out + f, a);
  }
  for (; f < n; ++f) {
    float s = bias;
    for (int i = 0; i < nin; ++i) s += w[i] * rows[i][f];
    out[f] = s;
  }
}

// y[j] = b[j] + sum_i x[i] * Wt[i][j], j < NOUT (multiple of 4): weights stored input-major so each input
// broadcasts against NOUT/4 accumulators
template <int NOUT>
inline void matvec_t(float* y, const float* b, const float* x, const float* Wt, int nin) {
  constexpr int NV = NOUT / 4;
  v4 acc[NV];
  for (int k = 0; k < NV; ++k) acc[k] = ld(b + 4 * k);
  for (int i = 0; i < nin; ++i) {
    const v4 xv = bc(x[i]);
    const float* w = Wt + i * NOUT;
    for (int k = 0; k < NV; ++k) acc[k] += xv * ld(w + 4 * k);
  }
  for (int k = 0; k < NV; ++k) sv(y + 4 * k, acc[k]);
}

// dot product, n multiple of 16
inline float dot16(const float* a, const float* b, int n) {
  v4 s0 = bc(0.0f), s1 = s0, s2 = s0, s3 = s0;
  for (int i = 0; i < n; i += 16) {
    s0 += ld(a + i) * ld(b + i);
    s1 += ld(a + i + 4) * ld(b + i + 4);
    s2 += ld(a + i + 8) * ld(b + i + 8);
    s3 += ld(a + i + 12) * ld(b + i + 12);
  }
  return hsum((s0 + s1) + (s2 + s3));
}

// ---------------------------------------------------------------------------------------------------------------
// weights
struct Tensor {
  std::vector<uint32_t> dims;
  std::vector<float> v;
};

std::vector<float> transpose(const float* a, int rows, int cols) {  // [rows][cols] -> [cols][rows]
  std::vector<float> t(rows * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) t[c * rows + r] = a[r * cols + c];
  return t;
}

struct GtBlock {  // one GTConv block (encoder or decoder; the decoder weights were pre-flipped by the exporter)
  const float *pc1w, *pc1b, *dww, *dwb, *pc2w, *pc2b, *gB, *fcb;
  float pc1a, dwa;
  std::vector<float> gWt, gRt, fcwt;  // GRU W^T [8][48], R^T [16][48], FC^T [16][8]
};

struct Gru {  // ONNX GRU weights, one direction, transposed: Wt [In][3H], Rt [H][3H]; B [6H] (gate order z, r, h)
  std::vector<float> Wt, Rt;
  const float* B = nullptr;
};

struct DpStage {
  Gru intra[2][2];  // [group][direction], hidden 4
  Gru inter[2];     // [group], hidden 8
  std::vector<float> intra_fct, inter_fct;  // [16 in][16 out]
  const float *intra_fcb, *intra_lnw, *intra_lnb, *inter_fcb, *inter_lnw, *inter_lnb;
};

struct Model {
  std::map<std::string, Tensor> t;
  const float *erb, *ierb, *c1w, *c1b, *c2w, *c2b, *d3b, *d4b;
  std::vector<float> d3w, d4w;  // regrouped per output channel, see load()
  float c1a, c2a, d3a;
  GtBlock enc[3], dec[3];
  DpStage dp[2];

  const float* get(const std::string& name, size_t n, std::string& err) {
    auto it = t.find(name);
    if (it == t.end()) {
      if (err.empty()) err = "missing weight " + name;
      return nullptr;
    }
    if (it->second.v.size() != n) {
      if (err.empty()) err = "weight " + name + " has " + std::to_string(it->second.v.size()) + " values, expected " + std::to_string(n);
      return nullptr;
    }
    return it->second.v.data();
  }
  float scalar(const std::string& name, std::string& err) {
    const float* p = get(name, 1, err);
    return p ? *p : 0.0f;
  }
  // ndir: directions stored in the record (2 for the bidirectional intra GRUs); dir: the one to take
  bool gru(Gru& g, const std::string& p, int in, int h, int ndir, int dir, std::string& err) {
    const float* W = get(p + "W", ndir * 3 * h * in, err);
    const float* R = get(p + "R", ndir * 3 * h * h, err);
    const float* B = get(p + "B", ndir * 6 * h, err);
    if (!W || !R || !B) return false;
    g.Wt = transpose(W + dir * 3 * h * in, 3 * h, in);
    g.Rt = transpose(R + dir * 3 * h * h, 3 * h, h);
    g.B = B + dir * 6 * h;
    return true;
  }

  bool load(const std::string& path, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
      err = path + ": cannot open";
      return false;
    }
    char magic[8];
    uint32_t n = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "GTCRNW01", 8) == 0 && std::fread(&n, 4, 1, f) == 1 && n < 1000;
    for (uint32_t i = 0; ok && i < n; ++i) {
      uint32_t len = 0, nd = 0;
      ok = std::fread(&len, 4, 1, f) == 1 && len < 256;
      if (!ok) break;
      std::string name(len, '\0');
      ok = std::fread(name.data(), 1, len, f) == len && std::fread(&nd, 4, 1, f) == 1 && nd <= 8;
      if (!ok) break;
      Tensor tt;
      tt.dims.resize(nd);
      ok = std::fread(tt.dims.data(), 4, nd, f) == nd;
      size_t total = 1;
      for (uint32_t d : tt.dims) total *= d;
      ok = ok && total < (1u << 24);
      if (!ok) break;
      tt.v.resize(total);
      ok = std::fread(tt.v.data(), 4, total, f) == total;
      t[name] = std::move(tt);
    }
    std::fclose(f);
    if (!ok) {
      err = path + ": not a GTCRNW01 weight file";
      return false;
    }
    erb = get("erb", kErb * kHigh, err);
    ierb = get("ierb", kHigh * kErb, err);
    c1w = get("c1w", 16 * 9 * 5, err);
    c1b = get("c1b", 16, err);
    c1a = scalar("c1a", err);
    c2w = get("c2w", 16 * 8 * 5, err);
    c2b = get("c2b", 16, err);
    c2a = scalar("c2a", err);
    for (int which = 0; which < 2; ++which) {
      for (int i = 0; i < 3; ++i) {
        const std::string p = std::string(which ? "dec" : "enc") + std::to_string(i) + ".";
        GtBlock& b = which ? dec[i] : enc[i];
        b.pc1w = get(p + "pc1w", 16 * 24, err);
        b.pc1b = get(p + "pc1b", 16, err);
        b.pc1a = scalar(p + "pc1a", err);
        b.dww = get(p + "dww", 16 * 9, err);
        b.dwb = get(p + "dwb", 16, err);
        b.dwa = scalar(p + "dwa", err);
        b.pc2w = get(p + "pc2w", 8 * 16, err);
        b.pc2b = get(p + "pc2b", 8, err);
        const float* gW = get(p + "gW", 48 * 8, err);
        const float* gR = get(p + "gR", 48 * 16, err);
        b.gB = get(p + "gB", 96, err);
        const float* fcw = get(p + "fcw", 8 * 16, err);
        b.fcb = get(p + "fcb", 8, err);
        if (!err.empty()) return false;
        b.gWt = transpose(gW, 48, 8);
        b.gRt = transpose(gR, 48, 16);
        b.fcwt = transpose(fcw, 8, 16);
      }
    }
    for (int s = 0; s < 2; ++s) {
      const std::string p = "dp" + std::to_string(s) + ".";
      DpStage& d = dp[s];
      for (int g = 0; g < 2; ++g) {
        for (int dir = 0; dir < 2; ++dir)
          if (!gru(d.intra[g][dir], p + "intra_g" + std::to_string(g + 1) + ".", 8, 4, 2, dir, err)) return false;
        if (!gru(d.inter[g], p + "inter_g" + std::to_string(g + 1) + ".", 8, 8, 1, 0, err)) return false;
      }
      const float* fc1 = get(p + "intra_fcw", 16 * 16, err);
      d.intra_fcb = get(p + "intra_fcb", 16, err);
      d.intra_lnw = get(p + "intra_lnw", 33 * 16, err);
      d.intra_lnb = get(p + "intra_lnb", 33 * 16, err);
      const float* fc2 = get(p + "inter_fcw", 16 * 16, err);
      d.inter_fcb = get(p + "inter_fcb", 16, err);
      d.inter_lnw = get(p + "inter_lnw", 33 * 16, err);
      d.inter_lnb = get(p + "inter_lnb", 33 * 16, err);
      if (!err.empty()) return false;
      d.intra_fct = transpose(fc1, 16, 16);
      d.inter_fct = transpose(fc2, 16, 16);
    }
    // transposed convs 33 -> 65 (16 -> 16, 2 groups) and 65 -> 129 (16 -> 2), kernel 5, stride 2. Stored
    // [in][out/group][5]; regroup to [out][in-of-group][5] so one output is one rows_mac call.
    const float* w3 = get("d3w", 16 * 8 * 5, err);
    d3b = get("d3b", 16, err);
    d3a = scalar("d3a", err);
    const float* w4 = get("d4w", 16 * 2 * 5, err);
    d4b = get("d4b", 2, err);
    if (!err.empty()) return false;
    d3w.resize(16 * 8 * 5);
    for (int o = 0; o < 16; ++o)
      for (int ii = 0; ii < 8; ++ii)
        for (int k = 0; k < 5; ++k) d3w[(o * 8 + ii) * 5 + k] = w3[(((o / 8) * 8 + ii) * 8 + (o % 8)) * 5 + k];
    d4w.resize(2 * 16 * 5);
    for (int o = 0; o < 2; ++o)
      for (int i = 0; i < 16; ++i)
        for (int k = 0; k < 5; ++k) d4w[(o * 16 + i) * 5 + k] = w4[(i * 2 + o) * 5 + k];
    return true;
  }
};

// ---------------------------------------------------------------------------------------------------------------
struct State {
  float conv[2][C][16][F2];  // [enc/dec][channel][time slot][freq]; block with dilation d owns slots [a, a+2d)
  float tra[2][3][C];
  float inter[2][F2][C];
};

struct Scratch {
  float feat[3][kBins];
  float fe[3][F0 + 2];      // ERB features, padded by 1
  float ev[C][F1 + 2], od[C][F1 + 2];  // even / odd samples of a padded row (stride-2 convs)
  float e0[C][F1];          // conv1 output (skip for de_convs[4])
  float e1[C][F2];          // conv2 output (skip for de_convs[3])
  float en[3][C][F2];       // encoder GT block outputs (skips for the decoder blocks)
  float x[C][F2], y[C][F2];  // current activations
  float xpad[8][F2 + 2];    // x1 padded for the SFE
  float h[C][F2];           // pc1 output
  float taps[3][F2 + 2];    // the 3 time taps of one channel's depthwise conv, padded in frequency
  float h2[8][F2];
  float xt[F2][C], x1[F2][C], tmp[F2][C];
  float gx[4][F2][12];      // intra GRU input projections, [chain][bin][gate]
  float gg[F2][24], gh[F2][24], sg[F2][24];
  float xp[C][F1 + 2];      // padded input of the transposed convs
  float d3[C][F1], m[2][F0];
};

void gt_block(const GtBlock& b, const float (*xin)[F2], float (*xout)[F2], float (*cache)[16][F2], int slot0, int dil,
              float* tra_h, Scratch& s) {
  // x1 = channels 0..7 (processed), x2 = channels 8..15 (passed through, interleaved at the end)
  const float* rows[24];
  for (int c = 0; c < 8; ++c) {
    s.xpad[c][0] = s.xpad[c][F2 + 1] = 0.0f;
    std::memcpy(&s.xpad[c][1], xin[c], F2 * sizeof(float));
    for (int k = 0; k < 3; ++k) rows[c * 3 + k] = &s.xpad[c][k];  // SFE: input channel c*3+k is x1[c][f+k-1]
  }
  for (int o = 0; o < C; ++o) {
    rows_mac(s.h[o], b.pc1b[o], rows, b.pc1w + o * 24, 24, F2);
    prelu_n(s.h[o], b.pc1a, F2);
  }
  // depthwise 3x3 (time x freq) conv over [cache slots..., new frame], dilation dil in time, pad 1 in freq
  const int T = 2 * dil;  // cached frames
  for (int c = 0; c < C; ++c) {
    for (int kt = 0; kt < 3; ++kt) {
      const float* src = kt < 2 ? cache[c][slot0 + kt * dil] : s.h[c];
      s.taps[kt][0] = s.taps[kt][F2 + 1] = 0.0f;
      std::memcpy(&s.taps[kt][1], src, F2 * sizeof(float));
      for (int kf = 0; kf < 3; ++kf) rows[kt * 3 + kf] = &s.taps[kt][kf];
    }
    rows_mac(s.y[c], b.dwb[c], rows, b.dww + c * 9, 9, F2);
    prelu_n(s.y[c], b.dwa, F2);
    // shift the cache: drop the oldest slot, append the new frame
    for (int t = 0; t + 1 < T; ++t) std::memcpy(cache[c][slot0 + t], cache[c][slot0 + t + 1], F2 * sizeof(float));
    std::memcpy(cache[c][slot0 + T - 1], s.h[c], F2 * sizeof(float));
  }
  // point conv 2: 16 -> 8
  for (int c = 0; c < C; ++c) rows[c] = s.y[c];
  for (int o = 0; o < 8; ++o) rows_mac(s.h2[o], b.pc2b[o], rows, b.pc2w + o * C, C, F2);
  // temporal recurrent attention: GRU(8 -> 16) on the per-channel energies, FC 16 -> 8, sigmoid gate
  float zt[8], g[48], gh[48], sg[48], at[8];
  for (int c = 0; c < 8; ++c) zt[c] = (dot16(s.h2[c], s.h2[c], 32) + s.h2[c][32] * s.h2[c][32]) * (1.0f / F2);
  matvec_t<48>(g, b.gB, zt, b.gWt.data(), 8);
  matvec_t<48>(gh, b.gB + 48, tra_h, b.gRt.data(), 16);
  for (int j = 0; j < 32; j += 4) sv(sg + j, sigmoid4(ld(g + j) + ld(gh + j)));  // z, r
  for (int j = 0; j < 16; j += 4) {
    const v4 z = ld(sg + j), r = ld(sg + 16 + j);
    const v4 nn = tanh4(ld(g + 32 + j) + r * ld(gh + 32 + j));
    sv(tra_h + j, (bc(1.0f) - z) * nn + z * ld(tra_h + j));
  }
  matvec_t<8>(at, b.fcb, tra_h, b.fcwt.data(), 16);
  for (int c = 0; c < 8; c += 4) sv(at + c, sigmoid4(ld(at + c)));
  for (int c = 0; c < 8; ++c) {
    float* out = xout[2 * c];
    for (int f = 0; f < F2; ++f) out[f] = s.h2[c][f] * at[c];
    std::memcpy(xout[2 * c + 1], xin[8 + c], F2 * sizeof(float));  // channel shuffle with x2
  }
}

void layer_norm_add(float (*x)[C], const float (*v)[C], const float* w, const float* bb) {
  // x += LN(v) over all 33*16 values, eps 1e-8, affine [33][16]
  const float* pv = &v[0][0];
  v4 s = bc(0.0f);
  for (int i = 0; i < F2 * C; i += 4) s += ld(pv + i);
  const float mean = hsum(s) * (1.0f / (F2 * C));
  const v4 mv = bc(mean);
  s = bc(0.0f);
  for (int i = 0; i < F2 * C; i += 4) {
    const v4 d = ld(pv + i) - mv;
    s += d * d;
  }
  const float var = hsum(s) * (1.0f / (F2 * C));
  const v4 inv = bc(1.0f / std::sqrt(var + 1e-8f));
  float* px = &x[0][0];
  for (int i = 0; i < F2 * C; i += 4) sv(px + i, ld(px + i) + (ld(pv + i) - mv) * inv * ld(w + i) + ld(bb + i));
}

void dpgrnn(const DpStage& d, float (*x)[F2], float (*icache)[C], Scratch& s) {
  // (C,F) -> (F,C)
  for (int f = 0; f < F2; ++f)
    for (int c = 0; c < C; ++c) s.xt[f][c] = x[c][f];
  // intra: per group a bidirectional GRU (8 -> 4) along frequency; output per bin = [g1 fwd, g1 bwd, g2 fwd, g2 bwd].
  // The 4 chains (2 groups x 2 directions) are independent: their input projections are batched and the 33
  // recurrence steps run the 4 chains side by side.
  const Gru* ch[4] = {&d.intra[0][0], &d.intra[0][1], &d.intra[1][0], &d.intra[1][1]};
  for (int k = 0; k < 4; ++k)
    for (int f = 0; f < F2; ++f) matvec_t<12>(s.gx[k][f], ch[k]->B, &s.xt[f][(k / 2) * 8], ch[k]->Wt.data(), 8);
  v4 h[4] = {bc(0.0f), bc(0.0f), bc(0.0f), bc(0.0f)};
  for (int i = 0; i < F2; ++i) {
    for (int k = 0; k < 4; ++k) {
      const int f = (k & 1) ? F2 - 1 - i : i;  // odd chains run backwards
      const float* Rt = ch[k]->Rt.data();      // [4][12]
      const float* Br = ch[k]->B + 12;
      v4 g0 = ld(Br), g1 = ld(Br + 4), g2 = ld(Br + 8);  // gh = Br + R h
      for (int j = 0; j < 4; ++j) {
        const v4 hv = bc(h[k][j]);
        g0 += hv * ld(Rt + j * 12);
        g1 += hv * ld(Rt + j * 12 + 4);
        g2 += hv * ld(Rt + j * 12 + 8);
      }
      const float* gx = s.gx[k][f];
      const v4 z = sigmoid4(ld(gx) + g0), r = sigmoid4(ld(gx + 4) + g1);
      const v4 nn = tanh4(ld(gx + 8) + r * g2);
      h[k] = (bc(1.0f) - z) * nn + z * h[k];
      sv(&s.tmp[f][k * 4], h[k]);
    }
  }
  for (int f = 0; f < F2; ++f) matvec_t<16>(s.x1[f], d.intra_fcb, s.tmp[f], d.intra_fct.data(), C);
  std::memcpy(s.tmp, s.xt, sizeof(s.xt));
  layer_norm_add(s.tmp, s.x1, d.intra_lnw, d.intra_lnb);  // tmp = x + LN(fc(intra))
  // inter: per group a GRU (8 -> 8) along time, one step, independent for every bin; its state is the cache
  for (int g = 0; g < 2; ++g) {
    const Gru& r = d.inter[g];
    for (int f = 0; f < F2; ++f) {
      matvec_t<24>(s.gg[f], r.B, &s.tmp[f][g * 8], r.Wt.data(), 8);
      matvec_t<24>(s.gh[f], r.B + 24, &icache[f][g * 8], r.Rt.data(), 8);
    }
    for (int f = 0; f < F2; ++f) {
      for (int j = 0; j < 16; j += 4) sv(&s.sg[f][j], sigmoid4(ld(&s.gg[f][j]) + ld(&s.gh[f][j])));
      float* hc = &icache[f][g * 8];
      for (int j = 0; j < 8; j += 4) {
        const v4 z = ld(&s.sg[f][j]), rr = ld(&s.sg[f][8 + j]);
        const v4 nn = tanh4(ld(&s.gg[f][16 + j]) + rr * ld(&s.gh[f][16 + j]));
        sv(hc + j, (bc(1.0f) - z) * nn + z * ld(hc + j));
      }
    }
  }
  for (int f = 0; f < F2; ++f) matvec_t<16>(s.x1[f], d.inter_fcb, icache[f], d.inter_fct.data(), C);
  layer_norm_add(s.tmp, s.x1, d.inter_lnw, d.inter_lnb);
  for (int f = 0; f < F2; ++f)
    for (int c = 0; c < C; ++c) x[c][f] = s.tmp[f][c];
}

// de-interleave a row padded by 2 on each side into even / odd sample rows (for the stride-2 convs):
// ev[j] = p[2j], od[j] = p[2j+1] with p = [0, 0, x[0..n-1], 0, 0]
inline void split_even_odd(const float* x, int n, float* ev, float* od) {
  const int half = n / 2 + 3;  // p has n + 4 samples (n odd): ev gets (n + 5) / 2 of them
  ev[0] = 0.0f;
  od[0] = 0.0f;
  for (int j = 1; j < half; ++j) {
    const int i = 2 * j - 2;  // index into x of p[2j]
    ev[j] = i < n ? x[i] : 0.0f;
    od[j] = i + 1 < n ? x[i + 1] : 0.0f;
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
struct GtcrnNative::Impl {
  Model m;
  State st{};
  Scratch s{};
  float mix[kBins * 2], enh[kBins * 2];
  float gain = 1.0f;
  bool loaded = false;
  double ms_sum = 0.0;
  int ms_n = 0;

  void step(const float* mix, float* out) {
    // features: |X|, Re, Im -> ERB band mapping (65 low bins kept, 192 high bins -> 64 bands)
    float(*feat)[kBins] = s.feat;
    for (int k = 0; k < kBins; ++k) {
      const float re = mix[2 * k], im = mix[2 * k + 1];
      feat[0][k] = std::sqrt(re * re + im * im + 1e-12f);
      feat[1][k] = re;
      feat[2][k] = im;
    }
    for (int c = 0; c < 3; ++c) {
      float* fe = s.fe[c];
      fe[0] = fe[F0 + 1] = 0.0f;
      std::memcpy(&fe[1], feat[c], kLow * sizeof(float));
      for (int e = 0; e < kErb; ++e) fe[1 + kLow + e] = dot16(m.erb + e * kHigh, feat[c] + kLow, kHigh);
    }
    const float* rows[80];
    // conv1: 9 -> 16, kernel 5, stride 2, pad 2 (129 -> 65). SFE input channel c*3+k is fe[c][f+k-1]; with the
    // row padded by 2 and split into even/odd samples, tap k of output f reads p[2f+k] = {ev, od, ev+1, od+1, ev+2}[k][f].
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k) {
        const int i = c * 3 + k;
        float p[F0 + 4];
        p[0] = p[1] = p[F0 + 2] = p[F0 + 3] = 0.0f;
        std::memcpy(p + 2, &s.fe[c][k], F0 * sizeof(float));
        for (int j = 0; j < F1 + 2; ++j) {
          s.ev[i][j] = 2 * j < F0 + 4 ? p[2 * j] : 0.0f;
          s.od[i][j] = 2 * j + 1 < F0 + 4 ? p[2 * j + 1] : 0.0f;
        }
        rows[i * 5] = s.ev[i];
        rows[i * 5 + 1] = s.od[i];
        rows[i * 5 + 2] = s.ev[i] + 1;
        rows[i * 5 + 3] = s.od[i] + 1;
        rows[i * 5 + 4] = s.ev[i] + 2;
      }
    for (int o = 0; o < C; ++o) {
      rows_mac(s.e0[o], m.c1b[o], rows, m.c1w + o * 45, 45, F1);
      prelu_n(s.e0[o], m.c1a, F1);
    }
    // conv2: 16 -> 16 in 2 groups of 8, kernel 5, stride 2, pad 2 (65 -> 33)
    for (int i = 0; i < C; ++i) {
      split_even_odd(s.e0[i], F1, s.ev[i], s.od[i]);
      rows[i * 5] = s.ev[i];
      rows[i * 5 + 1] = s.od[i];
      rows[i * 5 + 2] = s.ev[i] + 1;
      rows[i * 5 + 3] = s.od[i] + 1;
      rows[i * 5 + 4] = s.ev[i] + 2;
    }
    for (int o = 0; o < C; ++o) {
      rows_mac(s.e1[o], m.c2b[o], rows + (o / 8) * 40, m.c2w + o * 40, 40, F2);
      prelu_n(s.e1[o], m.c2a, F2);
    }
    // encoder GT blocks (dilations 1, 2, 5 own cache slots [0,2), [2,6), [6,16))
    static const int kSlot[3] = {0, 2, 6}, kDil[3] = {1, 2, 5};
    gt_block(m.enc[0], s.e1, s.en[0], st.conv[0], kSlot[0], kDil[0], st.tra[0][0], s);
    gt_block(m.enc[1], s.en[0], s.en[1], st.conv[0], kSlot[1], kDil[1], st.tra[0][1], s);
    gt_block(m.enc[2], s.en[1], s.en[2], st.conv[0], kSlot[2], kDil[2], st.tra[0][2], s);
    std::memcpy(s.x, s.en[2], sizeof(s.x));
    dpgrnn(m.dp[0], s.x, st.inter[0], s);
    dpgrnn(m.dp[1], s.x, st.inter[1], s);
    // decoder GT blocks with skip connections, mirrored dilations 5, 2, 1
    for (int i = 0; i < 3; ++i) {
      const float(*skip)[F2] = s.en[2 - i];
      for (int c = 0; c < C; ++c)
        for (int f = 0; f < F2; ++f) s.x[c][f] += skip[c][f];
      gt_block(m.dec[i], s.x, s.y, st.conv[1], kSlot[2 - i], kDil[2 - i], st.tra[1][i], s);
      std::memcpy(s.x, s.y, sizeof(s.x));
    }
    // transposed convs along frequency, kernel 5, stride 2, pad 2 (n -> 2n-1), on the input padded by one zero:
    //   out[2m]   = w4 x[m-1] + w2 x[m] + w0 x[m+1]     m = 0..n-1
    //   out[2m+1] = w3 x[m]   + w1 x[m+1]               m = 0..n-2
    // de_convs[3]: 16 -> 16 (2 groups), 33 -> 65, + PReLU, skip e0 (its input is x + e1)
    for (int c = 0; c < C; ++c) {
      s.xp[c][0] = s.xp[c][F2 + 1] = 0.0f;
      for (int f = 0; f < F2; ++f) s.xp[c][1 + f] = s.x[c][f] + s.e1[c][f];
    }
    for (int o = 0; o < C; ++o) {
      float we[24], wo[16], ev[F2], od[F2];
      const float* rowo[16];
      for (int ii = 0; ii < 8; ++ii) {
        const float* xp = s.xp[(o / 8) * 8 + ii];
        const float* w = m.d3w.data() + (o * 8 + ii) * 5;
        rows[ii * 3] = xp;
        rows[ii * 3 + 1] = xp + 1;
        rows[ii * 3 + 2] = xp + 2;
        we[ii * 3] = w[4];
        we[ii * 3 + 1] = w[2];
        we[ii * 3 + 2] = w[0];
        rowo[ii * 2] = xp + 1;
        rowo[ii * 2 + 1] = xp + 2;
        wo[ii * 2] = w[3];
        wo[ii * 2 + 1] = w[1];
      }
      rows_mac(ev, m.d3b[o], rows, we, 24, F2);
      rows_mac(od, m.d3b[o], rowo, wo, 16, F2 - 1);
      float* out = s.d3[o];
      for (int j = 0; j < F2; ++j) out[2 * j] = prelu(ev[j], m.d3a) + s.e0[o][2 * j];
      for (int j = 0; j + 1 < F2; ++j) out[2 * j + 1] = prelu(od[j], m.d3a) + s.e0[o][2 * j + 1];
    }
    // de_convs[4]: 16 -> 2, 65 -> 129, + tanh = complex mask on the ERB axis
    for (int c = 0; c < C; ++c) {
      s.xp[c][0] = s.xp[c][F1 + 1] = 0.0f;
      std::memcpy(&s.xp[c][1], s.d3[c], F1 * sizeof(float));
    }
    for (int o = 0; o < 2; ++o) {
      float we[48], wo[32], ev[F1 + 3], od[F1 + 3];
      const float* rowo[32];
      for (int i = 0; i < C; ++i) {
        const float* xp = s.xp[i];
        const float* w = m.d4w.data() + (o * 16 + i) * 5;
        rows[i * 3] = xp;
        rows[i * 3 + 1] = xp + 1;
        rows[i * 3 + 2] = xp + 2;
        we[i * 3] = w[4];
        we[i * 3 + 1] = w[2];
        we[i * 3 + 2] = w[0];
        rowo[i * 2] = xp + 1;
        rowo[i * 2 + 1] = xp + 2;
        wo[i * 2] = w[3];
        wo[i * 2 + 1] = w[1];
      }
      rows_mac(ev, m.d4b[o], rows, we, 48, F1);
      rows_mac(od, m.d4b[o], rowo, wo, 32, F1 - 1);
      for (int j = 0; j < F1 + 3; j += 4) sv(ev + j, tanh4(ld(ev + j)));  // (3 pad lanes are garbage, unused)
      for (int j = 0; j < F1 - 1; j += 4) sv(od + j, tanh4(ld(od + j)));
      float* mm = s.m[o];
      for (int j = 0; j < F1; ++j) mm[2 * j] = ev[j];
      for (int j = 0; j + 1 < F1; ++j) mm[2 * j + 1] = od[j];
    }
    // inverse ERB mapping of the mask (64 bands -> 192 bins) and complex multiply with the input
    for (int k = 0; k < kLow; ++k) {
      const float mr = s.m[0][k], mi = s.m[1][k], re = mix[2 * k], im = mix[2 * k + 1];
      out[2 * k] = re * mr - im * mi;
      out[2 * k + 1] = im * mr + re * mi;
    }
    for (int f = 0; f < kHigh; ++f) {
      const float* w = m.ierb + f * kErb;
      const float mr = dot16(w, s.m[0] + kLow, kErb), mi = dot16(w, s.m[1] + kLow, kErb);
      const int k = kLow + f;
      const float re = mix[2 * k], im = mix[2 * k + 1];
      out[2 * k] = re * mr - im * mi;
      out[2 * k + 1] = im * mr + re * mi;
    }
  }
};

GtcrnNative::GtcrnNative() : im_(new Impl) {}
GtcrnNative::~GtcrnNative() = default;

bool GtcrnNative::load(const std::string& path, float gain, std::string& err) {
  if (!im_->m.load(path, err)) return false;
  im_->gain = gain;
  im_->loaded = true;
  reset();
  return true;
}

void GtcrnNative::reset() { std::memset(&im_->st, 0, sizeof(im_->st)); }

void GtcrnNative::step(const float* mix, float* enh) { im_->step(mix, enh); }

void GtcrnNative::process(std::complex<float>* spec, int bins) {
  if (!im_->loaded || bins != kBins) return;
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < kBins; ++k) {
    im_->mix[2 * k] = spec[k].real() * im_->gain;
    im_->mix[2 * k + 1] = spec[k].imag() * im_->gain;
  }
  im_->step(im_->mix, im_->enh);
  const float inv = 1.0f / im_->gain;
  for (int k = 0; k < kBins; ++k) spec[k] = std::complex<float>(im_->enh[2 * k] * inv, im_->enh[2 * k + 1] * inv);
  im_->ms_sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ++im_->ms_n;
}

double GtcrnNative::avg_ms() const {
  const double v = im_->ms_n ? im_->ms_sum / im_->ms_n : 0.0;
  im_->ms_sum = 0.0;
  im_->ms_n = 0;
  return v;
}

const float* GtcrnNative::conv_cache() const { return &im_->st.conv[0][0][0][0]; }
const float* GtcrnNative::tra_cache() const { return &im_->st.tra[0][0][0]; }
const float* GtcrnNative::inter_cache() const { return &im_->st.inter[0][0][0]; }
