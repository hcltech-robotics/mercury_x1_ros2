#include "array.h"

#include <algorithm>
#include <cmath>

namespace {

// Solves A z = b for a small dense complex system (Gaussian elimination, partial pivoting). A is n*n row-major.
bool solve(std::vector<cf> a, std::vector<cf> b, int n, std::vector<cf>& z) {
  for (int c = 0; c < n; ++c) {
    int piv = c;
    for (int r = c + 1; r < n; ++r)
      if (std::abs(a[r * n + c]) > std::abs(a[piv * n + c])) piv = r;
    if (std::abs(a[piv * n + c]) < 1e-20f) return false;
    if (piv != c) {
      for (int k = 0; k < n; ++k) std::swap(a[c * n + k], a[piv * n + k]);
      std::swap(b[c], b[piv]);
    }
    for (int r = c + 1; r < n; ++r) {
      const cf f = a[r * n + c] / a[c * n + c];
      for (int k = c; k < n; ++k) a[r * n + k] -= f * a[c * n + k];
      b[r] -= f * b[c];
    }
  }
  z.assign(n, cf(0));
  for (int r = n - 1; r >= 0; --r) {
    cf s = b[r];
    for (int k = r + 1; k < n; ++k) s -= a[r * n + k] * z[k];
    z[r] = s / a[r * n + r];
  }
  return true;
}

}  // namespace

ArrayProcessor::ArrayProcessor(const ArrayConfig& cfg)
    : cfg_(cfg), m_(cfg.mics), fft_(kN), win_(kN), frame_(m_, std::vector<float>(kN, 0.0f)), pend_(m_),
      x_(m_, std::vector<cf>(kK)), work_(kN), ola_(kN, 0.0f), w_(m_, std::vector<cf>(kK)),
      rnn_(kK, std::vector<cf>(m_ * m_, cf(0))) {
  for (int n = 0; n < kN; ++n) win_[n] = std::sqrt(0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * n / kN));

  const float bin_hz = static_cast<float>(cfg_.rate) / kN;
  k_lo_ = std::max(1, static_cast<int>(std::lround(cfg_.doa_lo_hz / bin_hz)));
  k_hi_ = std::min(kK - 2, static_cast<int>(std::lround(cfg_.doa_hi_hz / bin_hz)));
  n_ang_ = 180 / cfg_.angle_step_deg + 1;
  for (int i = 0; i < m_; ++i)
    for (int j = i + 1; j < m_; ++j) pairs_.emplace_back(i, j);

  const int nb = k_hi_ - k_lo_ + 1, np = static_cast<int>(pairs_.size());
  srp_steer_.resize(static_cast<size_t>(n_ang_) * np * nb);
  for (int a = 0; a < n_ang_; ++a) {
    const float deg = static_cast<float>(a * cfg_.angle_step_deg);
    for (int p = 0; p < np; ++p) {
      const float dt = tau(pairs_[p].first, deg) - tau(pairs_[p].second, deg);
      for (int k = k_lo_; k <= k_hi_; ++k) {
        const float w = 2.0f * static_cast<float>(M_PI) * k * bin_hz;
        srp_steer_[(static_cast<size_t>(a) * np + p) * nb + (k - k_lo_)] = std::polar(1.0f, w * dt);
      }
    }
  }
  map_.assign(n_ang_, 0.0f);
  smooth_.assign(n_ang_, 0.0f);
  hist_.resize(std::max(1, static_cast<int>(cfg_.history_s * cfg_.rate / kH)));
  for (auto& h : hist_) h.map.assign(n_ang_, 0.0f);
  for (int k = 0; k < kK; ++k)
    for (int m = 0; m < m_; ++m) rnn_[k][m * m_ + m] = cf(1e-9f);
  update_weights();
}

float ArrayProcessor::tau(int m, float deg) const {
  const float x = (m - (m_ - 1) * 0.5f) * cfg_.pitch_m;
  return -x * std::cos(deg * static_cast<float>(M_PI) / 180.0f) / cfg_.sound_speed;
}

void ArrayProcessor::steer(float deg) {
  deg = std::clamp(deg, 0.0f, 180.0f);
  if (std::fabs(deg - steer_deg_) > 0.5f) {
    steer_deg_ = deg;
    w_dirty_ = true;
  }
}

void ArrayProcessor::set_beamformer(const std::string& kind) {
  cfg_.beamformer = kind == "mvdr" ? "mvdr" : "ds";
  w_dirty_ = true;
}

void ArrayProcessor::push(const float* const* mics, int n, bool far_active, std::vector<float>& out) {
  for (int m = 0; m < m_; ++m) pend_[m].insert(pend_[m].end(), mics[m], mics[m] + n);
  while (static_cast<int>(pend_[0].size()) >= kH) {
    for (int m = 0; m < m_; ++m) {
      auto& f = frame_[m];
      std::copy(f.begin() + kH, f.end(), f.begin());
      std::copy(pend_[m].begin(), pend_[m].begin() + kH, f.end() - kH);
      pend_[m].erase(pend_[m].begin(), pend_[m].begin() + kH);
    }
    samples_in_ += kH;
    process_frame(far_active);
    out.insert(out.end(), ola_.begin(), ola_.begin() + kH);
    std::copy(ola_.begin() + kH, ola_.end(), ola_.begin());
    std::fill(ola_.end() - kH, ola_.end(), 0.0f);
  }
}

void ArrayProcessor::process_frame(bool far_active) {
  for (int m = 0; m < m_; ++m) {
    for (int n = 0; n < kN; ++n) work_[n] = cf(frame_[m][n] * win_[n], 0.0f);
    fft_.forward(work_.data());
    std::copy(work_.begin(), work_.begin() + kK, x_[m].begin());
  }

  // voice activity: speech band energy against a noise floor that falls fast and rises ~0.8 dB/s
  double e = 0.0;
  for (int m = 0; m < m_; ++m)
    for (int k = k_lo_; k <= k_hi_; ++k) e += std::norm(x_[m][k]);
  const float ef = static_cast<float>(e / m_) + 1e-12f;
  if (floor_ < 0.0f) floor_ = ef;
  if (ef < floor_)
    floor_ = 0.7f * floor_ + 0.3f * ef;
  else
    floor_ = std::min(floor_ * 1.003f, ef);
  snr_db_ = 10.0f * std::log10(ef / floor_);
  if (snr_db_ > cfg_.vad_snr_db) {
    speech_ = true;
    hang_ = 10;
  } else if (hang_ > 0) {
    --hang_;
  } else {
    speech_ = false;
  }

  // SRP-PHAT map over the angle grid
  const int nb = k_hi_ - k_lo_ + 1, np = static_cast<int>(pairs_.size());
  std::vector<cf> g(static_cast<size_t>(np) * nb);
  for (int p = 0; p < np; ++p) {
    for (int k = k_lo_; k <= k_hi_; ++k) {
      const cf c = x_[pairs_[p].first][k] * std::conj(x_[pairs_[p].second][k]);
      const float mag = std::abs(c);
      g[static_cast<size_t>(p) * nb + (k - k_lo_)] = mag > 1e-20f ? c / mag : cf(0);
    }
  }
  const float norm = 1.0f / (np * nb);
  for (int a = 0; a < n_ang_; ++a) {
    float s = 0.0f;
    const cf* st = &srp_steer_[static_cast<size_t>(a) * np * nb];
    for (size_t i = 0; i < g.size(); ++i) s += g[i].real() * st[i].real() - g[i].imag() * st[i].imag();
    map_[a] = s * norm;
  }
  HistFrame& h = hist_[hist_pos_];
  h.end = samples_in_;
  h.speech = speech_;
  h.far = far_active;
  h.map = map_;
  hist_pos_ = (hist_pos_ + 1) % static_cast<int>(hist_.size());
  hist_len_ = std::min(hist_len_ + 1, static_cast<int>(hist_.size()));

  doa_fresh_ = false;
  if (speech_ && (!far_active || doa_during_far_)) {
    for (int a = 0; a < n_ang_; ++a) smooth_[a] = 0.8f * smooth_[a] + 0.2f * map_[a];
    peak(smooth_, doa_deg_, doa_conf_);
    doa_fresh_ = true;
  }

  // MVDR: noise covariance from frames without speech or playback
  if (cfg_.beamformer == "mvdr" && !speech_ && !far_active) {
    for (int k = 0; k < kK; ++k) {
      auto& r = rnn_[k];
      for (int i = 0; i < m_; ++i)
        for (int j = 0; j < m_; ++j) r[i * m_ + j] = 0.99f * r[i * m_ + j] + 0.01f * x_[i][k] * std::conj(x_[j][k]);
    }
    ++rnn_frames_;
    if (++frames_since_w_ >= 16) w_dirty_ = true;
  }
  if (w_dirty_) update_weights();

  for (int k = 0; k < kK; ++k) {
    cf y(0);
    for (int m = 0; m < m_; ++m) y += std::conj(w_[m][k]) * x_[m][k];
    work_[k] = y;
  }
  work_[0] = cf(work_[0].real(), 0.0f);
  work_[kN / 2] = cf(work_[kN / 2].real(), 0.0f);
  for (int k = 1; k < kN / 2; ++k) work_[kN - k] = std::conj(work_[k]);
  fft_.inverse(work_.data());
  for (int n = 0; n < kN; ++n) ola_[n] += work_[n].real() * win_[n];
}

void ArrayProcessor::update_weights() {
  w_dirty_ = false;
  frames_since_w_ = 0;
  const float bin_hz = static_cast<float>(cfg_.rate) / kN;
  std::vector<cf> d(m_), z;
  const bool mvdr = cfg_.beamformer == "mvdr" && rnn_frames_ >= 50;
  for (int k = 0; k < kK; ++k) {
    const float w = 2.0f * static_cast<float>(M_PI) * k * bin_hz;
    for (int m = 0; m < m_; ++m) d[m] = std::polar(1.0f, -w * tau(m, steer_deg_));
    bool done = false;
    if (mvdr && k > 0 && k < kK - 1) {
      std::vector<cf> r = rnn_[k];
      float tr = 0.0f;
      for (int m = 0; m < m_; ++m) tr += r[m * m_ + m].real();
      const float load = 0.05f * tr / m_ + 1e-12f;  // diagonal loading keeps it robust to steering errors
      for (int m = 0; m < m_; ++m) r[m * m_ + m] += load;
      if (solve(r, d, m_, z)) {
        cf den(0);
        for (int m = 0; m < m_; ++m) den += std::conj(d[m]) * z[m];
        if (std::abs(den) > 1e-20f) {
          for (int m = 0; m < m_; ++m) w_[m][k] = z[m] / std::conj(den);
          done = true;
        }
      }
    }
    if (!done)
      for (int m = 0; m < m_; ++m) w_[m][k] = d[m] / static_cast<float>(m_);
  }
}

void ArrayProcessor::peak(const std::vector<float>& map, float& deg, float& conf) const {
  int best = 0;
  float mn = map[0], mx = map[0], mean = 0.0f;
  for (int a = 0; a < n_ang_; ++a) {
    if (map[a] > mx) {
      mx = map[a];
      best = a;
    }
    mn = std::min(mn, map[a]);
    mean += map[a];
  }
  mean /= n_ang_;
  float off = 0.0f;
  if (best > 0 && best < n_ang_ - 1) {
    const float l = map[best - 1], r = map[best + 1], den = l - 2.0f * mx + r;
    if (den < -1e-12f) off = std::clamp(0.5f * (l - r) / den, -0.5f, 0.5f);
  }
  deg = std::clamp((best + off) * cfg_.angle_step_deg, 0.0f, 180.0f);
  conf = (mx - mean) / (mx - mn + 1e-9f);
}

bool ArrayProcessor::angle_between(int64_t s0, int64_t s1, float& deg, float& conf) const {
  std::vector<float> acc(n_ang_);
  for (int pass = 0; pass < 3; ++pass) {
    std::fill(acc.begin(), acc.end(), 0.0f);
    int used = 0;
    for (int i = 0; i < hist_len_; ++i) {
      const HistFrame& h = hist_[i];
      if (h.end < s0 || h.end > s1 + kN) continue;
      if (pass == 0 && !(h.speech && !h.far)) continue;
      if (pass == 1 && !h.speech) continue;
      for (int a = 0; a < n_ang_; ++a) acc[a] += h.map[a];
      ++used;
    }
    if (used > 0) {
      peak(acc, deg, conf);
      return true;
    }
  }
  return false;
}
