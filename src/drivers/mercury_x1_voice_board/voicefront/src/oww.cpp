#include "oww.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>

namespace {

constexpr int kBlock = 1280;      // 80 ms
constexpr int kMelCtx = 160 * 3;  // extra samples in front of every block (as openwakeword does)
constexpr int kMelBins = 32;
constexpr int kEmbWin = 76;       // mel frames per embedding
constexpr int kEmbDim = 96;
constexpr int kMelKeep = 10 * 97;
constexpr int kFeatKeep = 120;
constexpr int kWarmup = 5;        // first scores are forced to 0 (openwakeword does the same)

struct Net {
  std::unique_ptr<Ort::Session> s;
  std::string in, out;
  bool load(Ort::Env& env, const Ort::SessionOptions& so, const std::string& path, std::string& err) {
    try {
      s = std::make_unique<Ort::Session>(env, path.c_str(), so);
      Ort::AllocatorWithDefaultOptions a;
      in = s->GetInputNameAllocated(0, a).get();
      out = s->GetOutputNameAllocated(0, a).get();
    } catch (const Ort::Exception& e) {
      err = path + ": " + e.what();
      return false;
    }
    return true;
  }
  // runs on one float tensor of the given shape, returns the flat output
  std::vector<float> run(const float* x, const std::vector<int64_t>& shape) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    Ort::Value t = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(x), n, shape.data(), shape.size());
    const char* ins[] = {in.c_str()};
    const char* outs[] = {out.c_str()};
    auto r = s->Run(Ort::RunOptions{nullptr}, ins, &t, 1, outs, 1);
    const size_t m = r[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* p = r[0].GetTensorData<float>();
    return std::vector<float>(p, p + m);
  }
};

}  // namespace

struct OwwSpotter::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "voicefront"};
  Net mel, emb, cls;
  int n_feat = 16;  // embeddings the classifier takes (from its input shape)

  std::vector<float> raw;                   // int16-scaled samples, last kBlock + kMelCtx kept
  std::vector<float> pending;               // samples not yet forming a full block
  std::deque<std::array<float, kMelBins>> mels;
  std::deque<std::array<float, kEmbDim>> feats;
  int64_t fed = 0, blocks = 0, last_hit = -1;
  float peak = 0.0f, last = 0.0f;

  void push_mel(const std::vector<float>& x) {  // mel of x (int16 scale) appended, x/10 + 2
    const std::vector<float> m = mel.run(x.data(), {1, static_cast<int64_t>(x.size())});
    for (size_t i = 0; i + kMelBins <= m.size(); i += kMelBins) {
      std::array<float, kMelBins> row;
      for (int k = 0; k < kMelBins; ++k) row[k] = m[i + k] / 10.0f + 2.0f;
      mels.push_back(row);
    }
    while (static_cast<int>(mels.size()) > kMelKeep) mels.pop_front();
  }

  void push_embedding() {
    if (static_cast<int>(mels.size()) < kEmbWin) return;
    std::vector<float> w(static_cast<size_t>(kEmbWin) * kMelBins);
    size_t j = 0;
    for (size_t i = mels.size() - kEmbWin; i < mels.size(); ++i)
      for (int k = 0; k < kMelBins; ++k) w[j++] = mels[i][k];
    const std::vector<float> e = emb.run(w.data(), {1, kEmbWin, kMelBins, 1});
    std::array<float, kEmbDim> f{};
    for (int k = 0; k < kEmbDim && k < static_cast<int>(e.size()); ++k) f[k] = e[k];
    feats.push_back(f);
    while (static_cast<int>(feats.size()) > kFeatKeep) feats.pop_front();
  }

  float classify() {
    if (static_cast<int>(feats.size()) < n_feat) return 0.0f;
    std::vector<float> w(static_cast<size_t>(n_feat) * kEmbDim);
    size_t j = 0;
    for (size_t i = feats.size() - n_feat; i < feats.size(); ++i)
      for (int k = 0; k < kEmbDim; ++k) w[j++] = feats[i][k];
    const std::vector<float> out = cls.run(w.data(), {1, n_feat, kEmbDim});
    return out.empty() ? 0.0f : out[0];
  }

  // openwakeword starts with mel = ones and the embeddings of 4 s of faint noise
  void warm_start() {
    mels.assign(kEmbWin, [] { std::array<float, kMelBins> r; r.fill(1.0f); return r; }());
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> d(-1000, 999);
    std::vector<float> noise(16000 * 4);
    for (auto& v : noise) v = static_cast<float>(d(rng));
    const std::vector<float> m = mel.run(noise.data(), {1, static_cast<int64_t>(noise.size())});
    std::vector<std::array<float, kMelBins>> frames;
    for (size_t i = 0; i + kMelBins <= m.size(); i += kMelBins) {
      std::array<float, kMelBins> row;
      for (int k = 0; k < kMelBins; ++k) row[k] = m[i + k] / 10.0f + 2.0f;
      frames.push_back(row);
    }
    for (size_t s = 0; s + kEmbWin <= frames.size(); s += 8) {
      std::vector<float> w(static_cast<size_t>(kEmbWin) * kMelBins);
      size_t j = 0;
      for (size_t i = s; i < s + kEmbWin; ++i)
        for (int k = 0; k < kMelBins; ++k) w[j++] = frames[i][k];
      const std::vector<float> e = emb.run(w.data(), {1, kEmbWin, kMelBins, 1});
      std::array<float, kEmbDim> f{};
      for (int k = 0; k < kEmbDim && k < static_cast<int>(e.size()); ++k) f[k] = e[k];
      feats.push_back(f);
    }
  }
};

OwwSpotter::OwwSpotter() : im_(new Impl) {}
OwwSpotter::~OwwSpotter() = default;

bool OwwSpotter::init(const OwwConfig& cfg, std::string& err) {
  cfg_ = cfg;
  std::string dir = cfg.feature_dir;
  if (dir.empty()) {
    const size_t slash = cfg.model.find_last_of('/');
    dir = slash == std::string::npos ? "." : cfg.model.substr(0, slash);
  }
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(std::max(1, cfg.threads));
  so.SetInterOpNumThreads(1);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (!im_->mel.load(im_->env, so, dir + "/melspectrogram.onnx", err)) return false;
  if (!im_->emb.load(im_->env, so, dir + "/embedding_model.onnx", err)) return false;
  if (!im_->cls.load(im_->env, so, cfg.model, err)) return false;
  try {
    const auto shape = im_->cls.s->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 3 && shape[1] > 0) im_->n_feat = static_cast<int>(shape[1]);
    if (shape.size() != 3 || shape[2] != kEmbDim) {
      err = cfg.model + ": expected an input of shape [1, N, 96]";
      return false;
    }
  } catch (const Ort::Exception& e) {
    err = e.what();
    return false;
  }
  // phrase from the file name: "hey_mercury.onnx", "hey_jarvis_v0.1.onnx" -> "hey mercury", "hey jarvis"
  std::string name = cfg.model.substr(cfg.model.find_last_of('/') + 1);
  name = name.substr(0, name.rfind(".onnx"));
  const size_t v = name.rfind("_v");
  if (v != std::string::npos && v + 2 < name.size() && std::isdigit(static_cast<unsigned char>(name[v + 2])))
    name = name.substr(0, v);
  std::replace(name.begin(), name.end(), '_', ' ');
  keywords_ = {name};
  try {
    im_->warm_start();
  } catch (const Ort::Exception& e) {
    err = std::string("warm start: ") + e.what();
    return false;
  }
  std::fprintf(stderr, "voicefront: openwakeword model %s (\"%s\"), %d embeddings, threshold %.2f\n",
               cfg.model.c_str(), name.c_str(), im_->n_feat, cfg.threshold);
  return true;
}

bool OwwSpotter::set_keywords(const std::vector<std::string>& kw, float threshold, float, std::string& err) {
  for (const auto& k : kw) {
    std::string c;
    for (char ch : k) c += std::isalnum(static_cast<unsigned char>(ch)) ? static_cast<char>(std::tolower(ch)) : ' ';
    std::string want;
    for (char ch : c) if (ch != ' ' || (!want.empty() && want.back() != ' ')) want += ch;
    while (!want.empty() && want.back() == ' ') want.pop_back();
    if (want != keywords_[0]) {
      err = "this wake word model only knows \"" + keywords_[0] + "\" (train a model for another phrase)";
      return false;
    }
  }
  if (threshold > 0.0f && threshold < 1.0f) cfg_.threshold = threshold;
  return true;
}

float OwwSpotter::peak_score() {
  const float p = im_->peak;
  im_->peak = 0.0f;
  return p;
}

std::vector<KwsHit> OwwSpotter::feed(const float* x, int n) {
  std::vector<KwsHit> hits;
  Impl& m = *im_;
  for (int i = 0; i < n; ++i) m.pending.push_back(std::max(-32768.0f, std::min(32767.0f, x[i] * 32768.0f)));
  m.fed += n;
  while (static_cast<int>(m.pending.size()) >= kBlock) {
    m.raw.insert(m.raw.end(), m.pending.begin(), m.pending.begin() + kBlock);
    m.pending.erase(m.pending.begin(), m.pending.begin() + kBlock);
    if (static_cast<int>(m.raw.size()) > kBlock + kMelCtx)
      m.raw.erase(m.raw.begin(), m.raw.end() - (kBlock + kMelCtx));
    try {
      m.push_mel(m.raw);
      m.push_embedding();
      float s = m.classify();
      if (++m.blocks <= kWarmup) s = 0.0f;
      m.last = s;
      m.peak = std::max(m.peak, s);
      const int64_t at = m.fed - static_cast<int64_t>(m.pending.size());
      if (cfg_.trace) std::fprintf(stderr, "oww %.2f %.4f\n", static_cast<double>(at) / 16000.0, s);
      const bool quiet = m.last_hit < 0 || at - m.last_hit > static_cast<int64_t>(cfg_.refractory_s * 16000);
      if (s >= cfg_.threshold && quiet) {
        KwsHit h;
        h.keyword = keywords_[0];
        h.sample = at;
        h.duration_s = 1.0f;
        h.score = s;
        hits.push_back(h);
        m.last_hit = at;
      }
    } catch (const Ort::Exception& e) {
      std::fprintf(stderr, "voicefront: openwakeword: %s\n", e.what());
    }
  }
  return hits;
}
