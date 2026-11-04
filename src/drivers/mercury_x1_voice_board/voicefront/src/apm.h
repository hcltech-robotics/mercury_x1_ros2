// WebRTC audio processing (AEC3, noise suppression, AGC2) behind a small interface, so the WebRTC and
// abseil headers stay in one translation unit.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

// Echo cancellation on all mics against the speaker reference. 10 ms frames at 16 kHz.
class EchoCanceller {
 public:
  // export_linear: keep the output of the linear echo filter (before the non-linear suppressor) available.
  // tuning: "comms" (WebRTC default: echo gone, a talker over the robot is cut hard), "asr" (lets a talker
  // through during playback, for wake word / speech recognition barge-in), "light" (even more transparent).
  EchoCanceller(int mics, int delay_ms, bool export_linear, const std::string& tuning = "asr");
  ~EchoCanceller();
  // ref: 160 mono samples; mics: 160 * mics interleaved samples, processed in place.
  void process(const int16_t* ref, int16_t* mics);
  // Linear-filter output of the last frame (160 samples, mono, float in [-1, 1]); false if not available.
  bool linear(float* out) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Noise suppression and adaptive digital gain on the mono beam. 10 ms frames at 16 kHz.
class PostFilter {
 public:
  // ns_level: "off", "low", "moderate", "high", "veryhigh"
  PostFilter(const std::string& ns_level, bool agc);
  ~PostFilter();
  void process(int16_t* mono);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
