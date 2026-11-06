// GTCRN speech enhancement (Rong et al. 2024, ~24k parameters, trained on DNS3) on the beam's STFT: 512-sample
// frames, 256 hop, sqrt-Hann - the ArrayProcessor framing, so it runs between beamforming and synthesis. Streaming
// ONNX export from github.com/Xiaobin-Rong/gtcrn (stream/onnx_models/gtcrn_simple.onnx), caches carried frame to
// frame. Replaces WebRTC's spectral noise suppression: on the robot (fan hum ~250 Hz + hiss, talker 2 m away) it
// takes the noise floor down ~23 dB where WebRTC's "moderate" leaves the hum and pumps it with the AGC.
#pragma once

#include <complex>
#include <memory>
#include <string>

// Spectral noise suppressor on one 257-bin STFT frame, called once per hop between beamforming and synthesis.
class SpectralNs {
 public:
  virtual ~SpectralNs() = default;
  virtual void process(std::complex<float>* spec, int bins) = 0;  // in place; bins must be 257
  virtual double avg_ms() const = 0;                              // mean run time per frame since the last call
};

// GTCRN through ONNX Runtime (model/gtcrn.onnx). GtcrnNative in gtcrn_native.h is the same net hand-written.
class SpecDenoiser : public SpectralNs {
 public:
  SpecDenoiser();
  ~SpecDenoiser() override;
  // gain: applied to the spectrum before the net and removed after (the net expects speech near normal level;
  // the beam from mics at a low PGA sits ~20 dB under that)
  bool load(const std::string& path, float gain, int threads, std::string& err);
  void process(std::complex<float>* spec, int bins) override;
  double avg_ms() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> im_;
};
