// Hum filter for the mic inputs. The robot's torso puts a constant acoustic tone of ~250 Hz into all four mics: a
// cluster of lines 246..255 Hz (the tone and its +-2..4 Hz modulation sidebands), not one pure line. Measured
// 2026-11-05 it is 55..65 % of all noise power, and 93..97 % of the noise power is below 300 Hz. Two notches
// (centre -+3 Hz, 12 Hz wide each, 4th order in all) plus a 2nd-order 100 Hz high-pass take 6 dB out of the hum
// band and 4 dB out of the whole noise floor, for 1.5 dB of speech energy in 200..300 Hz (nothing above).
// Only the mics are filtered, never the reference: the AEC learns the echo path mic = H(ref), and a filter on the
// mic side just becomes part of H, while the same filter on the reference (even a 100 Hz high-pass alone) put
// near-empty bins into the render spectrum the adaptive filter normalises by, and echo suppression fell from
// 30 dB to 6. Measured on a recording: mics filtered, the robot's own voice in the output is 4 dB lower than
// without the filter. Also fixed on purpose: a tracking notch that followed the per-second peak of the cluster
// retuned every second, and each retune was a transient to the AEC.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

class Biquad {
 public:
  void set(float b0, float b1, float b2, float a1, float a2) { b0_ = b0; b1_ = b1; b2_ = b2; a1_ = a1; a2_ = a2; }
  inline float operator()(float x) {  // direct form II transposed
    const float y = b0_ * x + z1_;
    z1_ = b1_ * x - a1_ * y + z2_;
    z2_ = b2_ * x - a2_ * y;
    return y;
  }
  // notch at f0 with -3 dB bandwidth bw (Hz)
  void notch(float fs, float f0, float bw) {
    const float w = 2.0f * static_cast<float>(M_PI) * f0 / fs, r = std::exp(-static_cast<float>(M_PI) * bw / fs);
    const float c = std::cos(w);
    // zeros on the unit circle, poles at radius r: H = (1 - 2c z^-1 + z^-2) / (1 - 2rc z^-1 + r^2 z^-2), unity gain at dc
    const float g = (1.0f - 2.0f * r * c + r * r) / (2.0f - 2.0f * c);
    set(g, -2.0f * c * g, g, -2.0f * r * c, r * r);
  }
  // 2nd-order Butterworth high-pass
  void highpass(float fs, float fc) {
    const float k = std::tan(static_cast<float>(M_PI) * fc / fs), k2 = k * k, n = 1.0f / (1.0f + std::sqrt(2.0f) * k + k2);
    set(n, -2.0f * n, n, 2.0f * (k2 - 1.0f) * n, (1.0f - std::sqrt(2.0f) * k + k2) * n);
  }

 private:
  float b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0, z1_ = 0, z2_ = 0;
};

class HumFilter {
 public:
  // channels: filtered in place as interleaved int16. Notches at f0 -+ split, each bw wide; high-pass at hp
  HumFilter(int fs, int channels, float f0, float split = 3.0f, float bw = 12.0f, float hp = 100.0f)
      : f0_(f0), st_(channels) {
    for (auto& s : st_) {  // hp or bw <= 0: that stage is left as identity
      if (hp > 0) s.hp.highpass(static_cast<float>(fs), hp);
      if (bw > 0) {
        s.lo.notch(static_cast<float>(fs), f0 - split, bw);
        s.hi.notch(static_cast<float>(fs), f0 + split, bw);
      }
    }
  }
  float f0() const { return f0_; }

  void process(int16_t* x, int frames, int stride) {
    const int ch = static_cast<int>(st_.size());
    for (int i = 0; i < frames; ++i)
      for (int k = 0; k < ch; ++k) {
        auto& s = st_[k];
        const float v = s.hi(s.lo(s.hp(static_cast<float>(x[static_cast<size_t>(i) * stride + k]))));
        x[static_cast<size_t>(i) * stride + k] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, std::round(v))));
      }
  }

 private:
  struct Stage {
    Biquad hp, lo, hi;
  };
  float f0_;
  std::vector<Stage> st_;
};
