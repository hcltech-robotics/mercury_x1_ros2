// Small radix-2 complex FFT (power-of-two sizes). Enough for a 512-point STFT on four channels.
#pragma once

#include <cmath>
#include <complex>
#include <utility>
#include <vector>

using cf = std::complex<float>;

class Fft {
 public:
  explicit Fft(int n) : n_(n), rev_(n), tw_(n / 2) {
    int lg = 0;
    while ((1 << lg) < n) ++lg;
    for (int i = 0; i < n; ++i) {
      int r = 0;
      for (int b = 0; b < lg; ++b)
        if ((i >> b) & 1) r |= 1 << (lg - 1 - b);
      rev_[i] = r;
    }
    for (int i = 0; i < n / 2; ++i) tw_[i] = std::polar(1.0f, static_cast<float>(-2.0 * M_PI * i / n));
  }

  int size() const { return n_; }
  void forward(cf* a) const { run(a, false); }
  void inverse(cf* a) const {
    run(a, true);
    const float s = 1.0f / n_;
    for (int i = 0; i < n_; ++i) a[i] *= s;
  }

 private:
  void run(cf* a, bool inv) const {
    for (int i = 0; i < n_; ++i)
      if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
    for (int len = 2; len <= n_; len <<= 1) {
      const int step = n_ / len, half = len / 2;
      for (int i = 0; i < n_; i += len) {
        for (int j = 0; j < half; ++j) {
          const cf w = inv ? std::conj(tw_[j * step]) : tw_[j * step];
          const cf u = a[i + j];
          const cf v = a[i + j + half] * w;
          a[i + j] = u + v;
          a[i + j + half] = u - v;
        }
      }
    }
  }

  int n_;
  std::vector<int> rev_;
  std::vector<cf> tw_;
};
