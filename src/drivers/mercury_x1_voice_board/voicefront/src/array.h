// Microphone-array stage: STFT, voice activity, direction of arrival (SRP-PHAT) and a steered beamformer.
//
// Geometry: a linear array along x, mic m at x = (m - (M-1)/2) * pitch. Angles are measured from the +x axis
// (the side of the last mic): 0 deg = end-fire towards mic M-1, 90 deg = broadside, 180 deg = towards mic 0.
// A linear array cannot tell front from back, so every angle stands for a cone around the axis.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "fft.h"

class SpectralNs;

struct ArrayConfig {
  int mics = 4;
  float pitch_m = 0.035f;
  int rate = 16000;
  float sound_speed = 343.0f;
  float doa_lo_hz = 300.0f;   // the room hum (150-300 Hz) stays out of the direction estimate
  float doa_hi_hz = 3500.0f;
  int angle_step_deg = 2;
  float vad_snr_db = 8.0f;
  float history_s = 3.0f;     // per-frame direction maps kept for the wake-word angle
  std::string beamformer = "ds";  // "ds" (delay-and-sum) or "mvdr"
};

class ArrayProcessor {
 public:
  static constexpr int kN = 512;  // 32 ms frame
  static constexpr int kH = 256;  // 16 ms hop, 50 % overlap (sqrt-Hann analysis + synthesis)
  static constexpr int kK = kN / 2 + 1;

  explicit ArrayProcessor(const ArrayConfig& cfg);

  // Feeds n samples per mic (deinterleaved, float in [-1, 1]); appends the beam output (if any) to `out`.
  // `far_active`: the robot's speaker is playing (no direction updates, no noise statistics then).
  void push(const float* const* mics, int n, bool far_active, std::vector<float>& out);

  void steer(float deg);
  float steered() const { return steer_deg_; }
  float doa() const { return doa_deg_; }
  float doa_confidence() const { return doa_conf_; }
  bool speech() const { return speech_; }
  float snr_db() const { return snr_db_; }
  bool doa_fresh() const { return doa_fresh_; }  // the direction was updated by the last frame
  int64_t samples_in() const { return samples_in_; }
  // Output sample index j corresponds to input sample j - latency().
  static constexpr int latency() { return kN - kH; }

  // Direction of the speech between input samples [s0, s1]; false if the history holds nothing usable.
  bool angle_between(int64_t s0, int64_t s1, float& deg, float& conf) const;

  void set_beamformer(const std::string& kind);
  void set_denoiser(SpectralNs* d) { dn_ = d; }   // on the beam spectrum, before synthesis
  const std::string& beamformer() const { return cfg_.beamformer; }
  void set_vad_snr_db(float db) { cfg_.vad_snr_db = db; }
  void set_doa_during_playback(bool on) { doa_during_far_ = on; }

 private:
  struct HistFrame {
    int64_t end = 0;  // input sample index just after the frame
    bool speech = false;
    bool far = false;
    std::vector<float> map;
  };

  void process_frame(bool far_active);
  void update_weights();
  void peak(const std::vector<float>& map, float& deg, float& conf) const;
  float tau(int m, float deg) const;

  ArrayConfig cfg_;
  int m_;
  Fft fft_;
  std::vector<float> win_;
  std::vector<std::vector<float>> frame_;  // [mic][kN] latest analysis window
  std::vector<std::vector<float>> pend_;   // [mic] samples not yet in a frame
  std::vector<std::vector<cf>> x_;         // [mic][kK]
  std::vector<cf> work_;
  std::vector<float> ola_;

  // direction of arrival
  int k_lo_, k_hi_, n_ang_;
  std::vector<std::pair<int, int>> pairs_;
  std::vector<cf> srp_steer_;  // [angle][pair][k - k_lo]
  std::vector<float> map_, smooth_;
  std::vector<HistFrame> hist_;
  int hist_pos_ = 0, hist_len_ = 0;
  float doa_deg_ = 90.0f, doa_conf_ = 0.0f;
  bool doa_fresh_ = false, doa_during_far_ = false;

  // voice activity (band energy against a tracked noise floor)
  float floor_ = -1.0f, snr_db_ = 0.0f;
  bool speech_ = false;
  int hang_ = 0;

  // beamformer
  float steer_deg_ = 90.0f;
  std::vector<std::vector<cf>> w_;    // [mic][kK]
  std::vector<std::vector<cf>> rnn_;  // [kK][m*m] noise covariance (MVDR)
  int rnn_frames_ = 0, frames_since_w_ = 0;
  bool w_dirty_ = true;

  SpectralNs* dn_ = nullptr;
  int64_t samples_in_ = 0;
};
