#include "kws.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "sherpa-onnx/c-api/c-api.h"

namespace {

const SherpaOnnxKeywordSpotter* sp(const void* p) { return static_cast<const SherpaOnnxKeywordSpotter*>(p); }
const SherpaOnnxOnlineStream* st(const void* p) { return static_cast<const SherpaOnnxOnlineStream*>(p); }

std::string upper(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

std::string lower(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// "hey mercury" -> "HEY MERCURY" (letters, digits, apostrophes; single spaces)
std::string clean(const std::string& s) {
  std::string out;
  for (char c : upper(s)) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') {
      out += c;
    } else if (!out.empty() && out.back() != ' ') {
      out += ' ';
    }
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

}  // namespace

KeywordSpotter::~KeywordSpotter() {
  if (stream_) SherpaOnnxDestroyOnlineStream(st(stream_));
  if (spotter_) SherpaOnnxDestroyKeywordSpotter(sp(spotter_));
}

bool KeywordSpotter::bpe_encode(const std::string& text, std::string& tokens) const {
  // The model's sentencepiece model is a unigram LM: per word, pick the segmentation of "\u2581WORD" into
  // vocabulary pieces with the highest total log probability (Viterbi), as sentencepiece does.
  static const std::string kSpace = "\xe2\x96\x81";  // U+2581, sentencepiece word boundary
  tokens.clear();
  std::stringstream ss(text);
  std::string word;
  while (ss >> word) {
    std::vector<std::string> ch{kSpace};
    for (char c : word) ch.emplace_back(1, c);
    const int n = static_cast<int>(ch.size());
    std::vector<float> best(n + 1, -1e30f);
    std::vector<int> from(n + 1, -1);
    best[0] = 0.0f;
    for (int i = 0; i < n; ++i) {
      if (best[i] <= -1e29f) continue;
      std::string piece;
      for (int j = i; j < n && j < i + 24; ++j) {
        piece += ch[j];
        auto it = vocab_.find(piece);
        if (it != vocab_.end() && best[i] + it->second > best[j + 1]) {
          best[j + 1] = best[i] + it->second;
          from[j + 1] = i;
        }
      }
    }
    if (from[n] < 0) return false;  // a character the model does not know
    std::vector<std::string> pieces;
    for (int e = n; e > 0; e = from[e]) {
      std::string piece;
      for (int k = from[e]; k < e; ++k) piece += ch[k];
      pieces.push_back(piece);
    }
    for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) tokens += (tokens.empty() ? "" : " ") + *it;
  }
  return !tokens.empty();
}

std::string KeywordSpotter::keyword_text(const std::vector<std::string>& kw, float threshold, float score) const {
  // one keyword per line: "<BPE TOKENS> :<boost> #<threshold> @<DISPLAY_NAME>"
  std::string text;
  char tail[64];
  for (const auto& k : kw) {
    const std::string c = clean(k);
    std::string tok;
    if (c.empty() || !bpe_encode(c, tok)) return "";
    std::string tag = c;
    std::replace(tag.begin(), tag.end(), ' ', '_');
    std::snprintf(tail, sizeof(tail), " :%.2f #%.3f @", score, threshold);
    text += tok + tail + tag + "\n";
  }
  return text;
}

bool KeywordSpotter::init(const KwsConfig& cfg, std::string& err) {
  cfg_ = cfg;
  const std::string d = cfg.model_dir + "/";
  const std::string enc = d + "encoder.int8.onnx", dec = d + "decoder.int8.onnx", joi = d + "joiner.int8.onnx",
                    tok = d + "tokens.txt", bpe = d + "bpe.vocab";
  {
    std::ifstream f(bpe);
    std::string line;
    while (std::getline(f, line)) {
      const size_t tab = line.find('\t');
      if (tab != std::string::npos) vocab_[line.substr(0, tab)] = std::strtof(line.c_str() + tab + 1, nullptr);
    }
  }
  if (vocab_.empty()) {
    err = "cannot read " + bpe;
    return false;
  }
  const std::string kw = keyword_text(cfg.keywords, cfg.threshold, cfg.score);
  if (kw.empty()) {
    err = "no usable keywords (letters, digits and apostrophes only)";
    return false;
  }
  std::fprintf(stderr, "voicefront: keywords:\n%s", kw.c_str());

  SherpaOnnxKeywordSpotterConfig c;
  std::memset(&c, 0, sizeof(c));
  c.feat_config.sample_rate = 16000;
  c.feat_config.feature_dim = 80;
  c.model_config.transducer.encoder = enc.c_str();
  c.model_config.transducer.decoder = dec.c_str();
  c.model_config.transducer.joiner = joi.c_str();
  c.model_config.tokens = tok.c_str();
  c.model_config.num_threads = cfg.threads;
  c.model_config.provider = "cpu";
  c.max_active_paths = 4;
  c.num_trailing_blanks = 1;
  c.keywords_score = cfg.score;
  c.keywords_threshold = cfg.threshold;
  c.keywords_buf = kw.c_str();
  c.keywords_buf_size = static_cast<int32_t>(kw.size());
  spotter_ = SherpaOnnxCreateKeywordSpotter(&c);
  if (!spotter_) {
    err = "cannot create the keyword spotter (model files in " + cfg.model_dir + "?)";
    return false;
  }
  stream_ = SherpaOnnxCreateKeywordStream(sp(spotter_));
  if (!stream_) {
    err = "cannot create a keyword stream";
    return false;
  }
  return true;
}

bool KeywordSpotter::set_keywords(const std::vector<std::string>& kw, float threshold, float score,
                                  std::string& err) {
  const std::string text = keyword_text(kw, threshold, score);
  if (text.empty()) {
    err = "no usable keywords (letters, digits and apostrophes only)";
    return false;
  }
  const SherpaOnnxOnlineStream* s = SherpaOnnxCreateKeywordStreamWithKeywords(sp(spotter_), text.c_str());
  if (!s) {
    err = "keywords rejected (cannot tokenize)";
    return false;
  }
  SherpaOnnxDestroyOnlineStream(st(stream_));
  stream_ = s;
  cfg_.keywords.clear();
  for (const auto& k : kw)
    if (!clean(k).empty()) cfg_.keywords.push_back(lower(clean(k)));
  cfg_.threshold = threshold;
  cfg_.score = score;
  return true;
}

std::vector<KwsHit> KeywordSpotter::feed(const float* x, int n) {
  std::vector<KwsHit> hits;
  SherpaOnnxOnlineStreamAcceptWaveform(st(stream_), 16000, x, n);
  fed_ += n;
  while (SherpaOnnxIsKeywordStreamReady(sp(spotter_), st(stream_))) {
    SherpaOnnxDecodeKeywordStream(sp(spotter_), st(stream_));
    const SherpaOnnxKeywordResult* r = SherpaOnnxGetKeywordResult(sp(spotter_), st(stream_));
    if (r && r->keyword && r->keyword[0]) {
      KwsHit h;
      h.keyword = r->keyword;
      std::replace(h.keyword.begin(), h.keyword.end(), '_', ' ');
      h.keyword = lower(h.keyword);
      h.sample = fed_;
      if (r->count > 1 && r->timestamps) h.duration_s = r->timestamps[r->count - 1] - r->timestamps[0] + 0.2f;
      hits.push_back(h);
      SherpaOnnxResetKeywordStream(sp(spotter_), st(stream_));
    }
    if (r) SherpaOnnxDestroyKeywordResult(r);
  }
  return hits;
}
