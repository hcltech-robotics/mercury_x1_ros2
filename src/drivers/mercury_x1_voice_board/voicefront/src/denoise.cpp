#include "denoise.h"

#include <chrono>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {
constexpr int kBins = 257;
}

struct SpecDenoiser::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "voicefront-ns"};
  std::unique_ptr<Ort::Session> s;
  std::vector<std::string> in_names, out_names;
  std::vector<int64_t> shape_mix{1, kBins, 1, 2}, shape_conv{2, 1, 16, 16, 33}, shape_tra{2, 3, 1, 1, 16},
      shape_inter{2, 1, 33, 16};
  std::vector<float> mix, conv, tra, inter;
  float gain = 1.0f;
  double ms_sum = 0.0;
  int ms_n = 0;
};

SpecDenoiser::SpecDenoiser() : im_(new Impl) {}
SpecDenoiser::~SpecDenoiser() = default;

bool SpecDenoiser::load(const std::string& path, float gain, int threads, std::string& err) {
  try {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads < 1 ? 1 : threads);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    im_->s = std::make_unique<Ort::Session>(im_->env, path.c_str(), so);
    Ort::AllocatorWithDefaultOptions a;
    if (im_->s->GetInputCount() != 4 || im_->s->GetOutputCount() != 4) {
      err = path + ": expected 4 inputs (mix + 3 caches) and 4 outputs";
      return false;
    }
    for (size_t i = 0; i < 4; ++i) {
      im_->in_names.push_back(im_->s->GetInputNameAllocated(i, a).get());
      im_->out_names.push_back(im_->s->GetOutputNameAllocated(i, a).get());
    }
    const auto sh = im_->s->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (sh.size() != 4 || sh[1] != kBins || sh[3] != 2) {
      err = path + ": expected a [1, 257, 1, 2] spectrum input";
      return false;
    }
  } catch (const Ort::Exception& e) {
    err = path + ": " + e.what();
    return false;
  }
  im_->gain = gain;
  im_->mix.assign(kBins * 2, 0.0f);
  im_->conv.assign(2 * 16 * 16 * 33, 0.0f);
  im_->tra.assign(2 * 3 * 16, 0.0f);
  im_->inter.assign(2 * 33 * 16, 0.0f);
  return true;
}

void SpecDenoiser::process(std::complex<float>* spec, int bins) {
  if (!im_->s || bins != kBins) return;
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < kBins; ++k) {
    im_->mix[k * 2] = spec[k].real() * im_->gain;
    im_->mix[k * 2 + 1] = spec[k].imag() * im_->gain;
  }
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value ins[4] = {
      Ort::Value::CreateTensor<float>(mem, im_->mix.data(), im_->mix.size(), im_->shape_mix.data(), 4),
      Ort::Value::CreateTensor<float>(mem, im_->conv.data(), im_->conv.size(), im_->shape_conv.data(), 5),
      Ort::Value::CreateTensor<float>(mem, im_->tra.data(), im_->tra.size(), im_->shape_tra.data(), 5),
      Ort::Value::CreateTensor<float>(mem, im_->inter.data(), im_->inter.size(), im_->shape_inter.data(), 4)};
  const char* in[4];
  const char* out[4];
  for (int i = 0; i < 4; ++i) {
    in[i] = im_->in_names[i].c_str();
    out[i] = im_->out_names[i].c_str();
  }
  try {
    auto r = im_->s->Run(Ort::RunOptions{nullptr}, in, ins, 4, out, 4);
    const float* enh = r[0].GetTensorData<float>();
    const float inv = 1.0f / im_->gain;
    for (int k = 0; k < kBins; ++k) spec[k] = std::complex<float>(enh[k * 2] * inv, enh[k * 2 + 1] * inv);
    std::vector<float>* caches[3] = {&im_->conv, &im_->tra, &im_->inter};
    for (int i = 0; i < 3; ++i) {
      const float* p = r[i + 1].GetTensorData<float>();
      caches[i]->assign(p, p + caches[i]->size());
    }
  } catch (const Ort::Exception&) {
    // a failed frame passes through unchanged
  }
  im_->ms_sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ++im_->ms_n;
}

double SpecDenoiser::avg_ms() const {
  const double v = im_->ms_n ? im_->ms_sum / im_->ms_n : 0.0;
  im_->ms_sum = 0.0;
  im_->ms_n = 0;
  return v;
}
