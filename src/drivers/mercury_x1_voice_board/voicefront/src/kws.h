// English open-vocabulary keyword spotting with sherpa-onnx (streaming zipformer transducer, BPE units).
// Keywords are plain text ("HEY MERCURY"); the model's piece vocabulary (bpe.vocab) tokenizes them at load time.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct KwsConfig {
  std::string model_dir;              // encoder/decoder/joiner *.int8.onnx, tokens.txt, bpe.vocab
  std::vector<std::string> keywords;  // plain text, any case
  float threshold = 0.25f;            // trigger threshold (lower = more sensitive)
  float score = 1.0f;                 // boosting score of the keyword paths
  int threads = 1;
};

struct KwsHit {
  std::string keyword;    // lower case, as configured
  int64_t sample = 0;     // samples fed when the hit was decoded
  float duration_s = 0;   // from the token time stamps (0 if unknown)
  float score = 0;        // detector score 0..1 when the engine has one (openwakeword), else 0
};

// What main.cpp needs from a wake-word engine (sherpa-onnx open-vocabulary KWS, or an openWakeWord model).
class WakeSpotter {
 public:
  virtual ~WakeSpotter() = default;
  virtual bool set_keywords(const std::vector<std::string>& kw, float threshold, float score, std::string& err) = 0;
  // Feeds 16 kHz float samples in [-1, 1]; returns the hits decoded so far.
  virtual std::vector<KwsHit> feed(const float* x, int n) = 0;
  virtual const std::vector<std::string>& keywords() const = 0;
  virtual float threshold() const = 0;
  virtual float score() const = 0;
  virtual const char* engine() const = 0;
  virtual float peak_score() { return 0.0f; }  // highest score since the last call, if the engine scores
};

class KeywordSpotter : public WakeSpotter {
 public:
  KeywordSpotter() = default;
  ~KeywordSpotter() override;
  KeywordSpotter(const KeywordSpotter&) = delete;
  KeywordSpotter& operator=(const KeywordSpotter&) = delete;

  bool init(const KwsConfig& cfg, std::string& err);
  // Replaces the keyword list (new stream); false (old list kept) if the text cannot be tokenized.
  bool set_keywords(const std::vector<std::string>& kw, float threshold, float score, std::string& err) override;
  std::vector<KwsHit> feed(const float* x, int n) override;
  const std::vector<std::string>& keywords() const override { return cfg_.keywords; }
  float threshold() const override { return cfg_.threshold; }
  float score() const override { return cfg_.score; }
  const char* engine() const override { return "sherpa"; }

 private:
  std::string keyword_text(const std::vector<std::string>& kw, float threshold, float score) const;
  // sentencepiece unigram tokenization: "HEY MERCURY" -> "\u2581HE Y \u2581ME R C UR Y" 
  bool bpe_encode(const std::string& text, std::string& tokens) const;

  std::unordered_map<std::string, float> vocab_;

  KwsConfig cfg_;
  const void* spotter_ = nullptr;
  const void* stream_ = nullptr;
  int64_t fed_ = 0;
};
