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
};

class KeywordSpotter {
 public:
  KeywordSpotter() = default;
  ~KeywordSpotter();
  KeywordSpotter(const KeywordSpotter&) = delete;
  KeywordSpotter& operator=(const KeywordSpotter&) = delete;

  bool init(const KwsConfig& cfg, std::string& err);
  // Replaces the keyword list (new stream); false (old list kept) if the text cannot be tokenized.
  bool set_keywords(const std::vector<std::string>& kw, float threshold, float score, std::string& err);
  // Feeds 16 kHz float samples; returns the hits decoded so far.
  std::vector<KwsHit> feed(const float* x, int n);
  const std::vector<std::string>& keywords() const { return cfg_.keywords; }
  float threshold() const { return cfg_.threshold; }
  float score() const { return cfg_.score; }

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
