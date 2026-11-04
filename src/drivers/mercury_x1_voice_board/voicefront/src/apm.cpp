#include "apm.h"

#include "api/audio/audio_processing.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/scoped_refptr.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"  // internal header (WebRTC source tree)

#include <array>
#include <vector>

namespace {
constexpr int kRate = 16000;

// AEC3 with our own configuration. The stock APM (v2.1) never sets filter.export_linear_aec_output, so asking
// it for the linear output dereferences a null framer; creating AEC3 ourselves fixes that and is where AEC3
// tuning goes.
class Aec3Factory : public webrtc::EchoControlFactory {
 public:
  Aec3Factory(bool export_linear, std::string tuning) : export_linear_(export_linear), tuning_(std::move(tuning)) {}
  std::unique_ptr<webrtc::EchoControl> Create(int rate, int num_render, int num_capture) override {
    using T = webrtc::EchoCanceller3Config::Suppressor::Tuning;
    using M = webrtc::EchoCanceller3Config::Suppressor::MaskingThresholds;
    webrtc::EchoCanceller3Config c;
    c.filter.export_linear_aec_output = export_linear_;
    auto& s = c.suppressor;
    if (tuning_ == "asr") {
      // enter the near-end (talker dominant) state at a lower SNR, and suppress less in both states
      s.dominant_nearend_detection.snr_threshold = 10.0f;
      s.dominant_nearend_detection.enr_threshold = 0.5f;
      s.dominant_nearend_detection.trigger_threshold = 6;
      s.normal_tuning = T(M(0.6f, 0.8f, 0.3f), M(0.15f, 0.3f, 0.3f), 2.0f, 0.25f);
      s.nearend_tuning = T(M(2.0f, 2.2f, 0.3f), M(0.6f, 1.0f, 0.3f), 2.0f, 0.25f);
    } else if (tuning_ == "light") {
      s.dominant_nearend_detection.snr_threshold = 6.0f;
      s.dominant_nearend_detection.enr_threshold = 1.0f;
      s.dominant_nearend_detection.trigger_threshold = 4;
      s.normal_tuning = T(M(1.09f, 1.1f, 0.3f), M(0.3f, 0.6f, 0.3f), 2.0f, 0.25f);
      s.nearend_tuning = T(M(3.0f, 3.5f, 0.3f), M(1.0f, 1.5f, 0.3f), 2.0f, 0.25f);
      c.ep_strength.default_gain = 0.5f;
    }
    return std::make_unique<webrtc::EchoCanceller3>(c, std::nullopt, rate, num_render, num_capture);
  }

 private:
  bool export_linear_;
  std::string tuning_;
};
}  // namespace

struct EchoCanceller::Impl {
  rtc::scoped_refptr<webrtc::AudioProcessing> apm;
  webrtc::StreamConfig ref_cfg{kRate, 1};
  webrtc::StreamConfig mic_cfg;
  int delay_ms = 0;
  int mics = 1;
};

EchoCanceller::EchoCanceller(int mics, int delay_ms, bool export_linear, const std::string& tuning)
    : impl_(new Impl) {
  webrtc::AudioProcessing::Config c;
  c.pipeline.multi_channel_capture = true;  // keep all mics (and their phase) for the beamformer
  c.pipeline.maximum_internal_processing_rate = 32000;
  c.echo_canceller.enabled = true;
  c.echo_canceller.mobile_mode = false;
  c.echo_canceller.export_linear_aec_output = export_linear;
  c.high_pass_filter.enabled = true;
  impl_->apm = webrtc::AudioProcessingBuilder()
                   .SetConfig(c)
                   .SetEchoControlFactory(std::make_unique<Aec3Factory>(export_linear, tuning))
                   .Create();
  impl_->mic_cfg = webrtc::StreamConfig(kRate, mics);
  impl_->delay_ms = delay_ms;
  impl_->mics = mics;
}

EchoCanceller::~EchoCanceller() = default;

void EchoCanceller::process(const int16_t* ref, int16_t* mics) {
  int16_t ref_out[160];
  impl_->apm->ProcessReverseStream(ref, impl_->ref_cfg, impl_->ref_cfg, ref_out);
  impl_->apm->set_stream_delay_ms(impl_->delay_ms);
  impl_->apm->ProcessStream(mics, impl_->mic_cfg, impl_->mic_cfg, mics);
}

bool EchoCanceller::linear(float* out) const {
  // one 160-sample frame per capture channel; channel 0 is used
  std::vector<std::array<float, 160>> buf(impl_->mics);
  if (!impl_->apm->GetLinearAecOutput(rtc::ArrayView<std::array<float, 160>>(buf.data(), buf.size()))) return false;
  for (int i = 0; i < 160; ++i) out[i] = buf[0][i];  // already converted to [-1, 1]
  return true;
}

struct PostFilter::Impl {
  rtc::scoped_refptr<webrtc::AudioProcessing> apm;
  webrtc::StreamConfig cfg{kRate, 1};
};

PostFilter::PostFilter(const std::string& ns_level, bool agc) : impl_(new Impl) {
  webrtc::AudioProcessing::Config c;
  c.pipeline.maximum_internal_processing_rate = 32000;
  using L = webrtc::AudioProcessing::Config::NoiseSuppression::Level;
  c.noise_suppression.enabled = ns_level != "off";
  c.noise_suppression.level = ns_level == "low"        ? L::kLow
                              : ns_level == "high"     ? L::kHigh
                              : ns_level == "veryhigh" ? L::kVeryHigh
                                                       : L::kModerate;
  c.gain_controller2.enabled = agc;
  c.gain_controller2.adaptive_digital.enabled = agc;
  impl_->apm = webrtc::AudioProcessingBuilder().SetConfig(c).Create();
}

PostFilter::~PostFilter() = default;

void PostFilter::process(int16_t* mono) { impl_->apm->ProcessStream(mono, impl_->cfg, impl_->cfg, mono); }
