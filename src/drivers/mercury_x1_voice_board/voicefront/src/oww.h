// Wake word with an openWakeWord model (https://github.com/dscripka/openWakeWord): a small classifier trained
// for one phrase, on Google speech-embedding features. Far more sensitive than open-vocabulary spotting, but
// the phrase is fixed at training time. Runs on ONNX Runtime (already linked for sherpa-onnx).
//
// Streaming is a port of openwakeword.utils.AudioFeatures: every 1280 samples (80 ms) the melspectrogram
// of the last 1280 + 480 samples is appended (x/10 + 2), the embedding of the last 76 mel frames is appended,
// and the classifier sees the last 16 embeddings (1.28 s + 0.76 s of context).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kws.h"

struct OwwConfig {
  std::string model;        // <phrase>.onnx, e.g. model/hey_mercury.onnx; the phrase is the file name
  std::string feature_dir;  // melspectrogram.onnx + embedding_model.onnx (default: next to the model)
  float threshold = 0.5f;   // score 0..1 that counts as a wake
  float refractory_s = 1.5f;
  int threads = 1;
  bool trace = false;       // print every block's score to stderr
};

class OwwSpotter : public WakeSpotter {
 public:
  OwwSpotter();
  ~OwwSpotter() override;
  bool init(const OwwConfig& cfg, std::string& err);

  bool set_keywords(const std::vector<std::string>& kw, float threshold, float score, std::string& err) override;
  std::vector<KwsHit> feed(const float* x, int n) override;
  const std::vector<std::string>& keywords() const override { return keywords_; }
  float threshold() const override { return cfg_.threshold; }
  float score() const override { return 0.0f; }
  const char* engine() const override { return "oww"; }
  float peak_score() override;  // thread-safe

 private:
  struct Impl;
  std::unique_ptr<Impl> im_;
  OwwConfig cfg_;
  std::vector<std::string> keywords_;
};
