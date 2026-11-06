// Hand-written streaming forward pass of GTCRN (the gtcrn_simple.onnx graph: ERB band mapping, SFE, 2 strided
// convs, 3 encoder GTConv blocks with temporal recurrent attention, 2 DPGRNN stages, mirrored decoder, complex
// mask). Same numbers as the ONNX Runtime path in denoise.cpp (SNR vs ORT > 60 dB) at a fraction of the cost: the
// ONNX graph is 415 tiny nodes and ORT's per-node dispatch alone takes ~4 ms on the voice board's Cortex-A53; this
// runs the ~0.5 M MACs directly in ~1 ms there. Weights: model/gtcrn.bin, written from the ONNX file by
// test/gtcrn_export.py (BatchNorm folded, decoder ConvTranspose weights flipped into plain causal convs).
#pragma once

#include <complex>
#include <memory>
#include <string>

#include "denoise.h"

class GtcrnNative : public SpectralNs {
 public:
  GtcrnNative();
  ~GtcrnNative() override;
  // gain: applied to the spectrum before the net and removed after (see SpecDenoiser::load)
  bool load(const std::string& path, float gain, std::string& err);
  void process(std::complex<float>* spec, int bins) override;  // in place; bins must be 257
  double avg_ms() const override;

  // One raw step for tests: mix / enh are [257][2] (real, imag), the recurrent state is kept inside.
  void step(const float* mix, float* enh);
  void reset();
  // state in the ONNX layout: conv_cache [2][16][16][33], tra_cache [2][3][16], inter_cache [2][33][16]
  const float* conv_cache() const;
  const float* tra_cache() const;
  const float* inter_cache() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> im_;
};
