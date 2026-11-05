// voicefront: microphone-array front end for the Mercury X1 voice board.
//
//   arecord -t raw -f S16_LE -r 16000 -c C | voicefront [options] | aplay -t raw -f S16_LE -r 16000 -c 1
//
// stdin : interleaved S16_LE frames with C channels (4 mics + the speaker echo reference among them)
// stdout: the processed beam, mono S16_LE 16 kHz (echo cancelled, beamformed, noise suppressed, AGC)
// link  : JSON lines (wake word, direction, voice activity, status; commands back), see README.md
//
// Pipeline per 10 ms: WebRTC AEC3 on every mic against the reference -> STFT (512/256) -> voice activity,
// SRP-PHAT direction, steered beamformer -> WebRTC noise suppression + AGC2 -> output + keyword spotter->

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "apm.h"
#include "array.h"
#include "json.h"
#include "kws.h"
#include "link.h"
#include "oww.h"

namespace {

constexpr int kRate = 16000;
constexpr int kChunk = 160;  // 10 ms
const char* kVersion = "0.1.0";

struct Options {
  int channels = 6;
  std::vector<int> mics{0, 1, 2, 3};
  int ref = 4;  // -1: no reference (no echo cancellation)
  bool aec = true;
  int aec_delay_ms = 0;
  std::string aec_tuning = "asr";
  float pitch_mm = 35.0f;
  std::string beamformer = "ds";
  float vad_snr_db = 8.0f;
  std::string ns = "moderate";
  bool agc = true;
  bool kws = true;
  bool kws_agc = false;        // AGC on the wake-word input (no gain in recall)
  std::string kws_ns = "off";  // noise suppression on the wake-word input (hurts recall in noise)
  float kws_gain_db = 0.0f;    // fixed gain on the wake-word input (the mics run at low PGA for headroom)
  std::string kws_input = "mic";  // beam | mic | linear (wake word listens to this signal)
  std::string model_dir = "model";
  std::vector<std::string> keywords{"hey mercury"};
  float kw_threshold = -1.0f;  // engine default: 0.25 sherpa, 0.5 openwakeword
  float kw_score = 2.0f;
  std::string oww;             // openWakeWord model file: switches the engine from sherpa KWS to it
  bool kws_trace = false;
  int kws_threads = 1;
  int sync_kws = -1;  // -1: synchronous when stdin is a regular file
  std::string serial;
  int baud = 115200;
  float doa_hz = 5.0f;
  float status_s = 5.0f;
  float wake_lock_s = 4.0f;
  bool doa_during_playback = false;
  std::string amixer_card;  // e.g. hw:sndac10710036 (enables set_gain)
  std::string dump;         // AEC output (mics) + reference, interleaved S16
  bool output = true;
  std::string out_cmd;  // beam to this command's stdin (restarted when it exits) instead of stdout
};

void usage() {
  std::fprintf(stderr,
               "voicefront %s - mic-array front end (AEC, DOA, beamforming, English wake word)\n"
               "  --channels N        input channels (default 6)\n"
               "  --mics a,b,c,d      mic channel indices, in array order (default 0,1,2,3)\n"
               "  --ref N             speaker reference channel, -1 = none (default 4)\n"
               "  --no-aec            skip echo cancellation\n"
               "  --aec-delay-ms N    reference-to-echo delay hint (default 0)\n"
               "  --aec-tuning comms|asr|light  echo suppression vs. talker over the robot (default asr)\n"
               "  --pitch-mm F        mic spacing (default 35)\n"
               "  --beamformer ds|mvdr  (default ds)\n"
               "  --vad-snr-db F      voice activity threshold (default 8)\n"
               "  --ns off|low|moderate|high|veryhigh  (default moderate)\n"
               "  --no-agc\n"
               "  --model DIR         keyword model directory (default ./model)\n"
               "  --keywords \"a;b\"    wake words, plain English (default \"hey mercury\"; sherpa engine)\n"
               "  --oww FILE.onnx     openWakeWord model for one trained phrase (the file name), replaces the\n"
               "                      sherpa engine; melspectrogram.onnx + embedding_model.onnx next to it or in --model\n"
               "  --kw-threshold F    lower = more sensitive (default 0.25 sherpa, 0.5 openwakeword)\n"
               "  --kw-score F        (default 2.0, sherpa only)\n"
               "  --kws-trace         print every wake-word score to stderr\n"
               "  --no-kws\n"
               "  --kws-input mic|beam|linear  wake-word input: echo-cancelled mic 0 (default, hears every\n"
               "                      direction), the steered beam, or the linear AEC output\n"
               "  --kws-agc 0|1       automatic gain on the wake-word input (default 0)\n"
               "  --kws-ns LEVEL      noise suppression on the wake-word input (default off)\n"
               "  --kws-gain-db F     fixed gain on the wake-word input, clipped (default 0)\n"
               "  --sync-kws 0|1      run the keyword spotter in the audio thread (default: 1 for file input)\n"
               "  --serial DEV        JSON-lines link (tty); default: events on stderr\n"
               "  --baud N            (default 115200)\n"
               "  --doa-hz F          direction events per second during speech, 0 = off (default 5)\n"
               "  --status-s F        status interval, 0 = off (default 5)\n"
               "  --wake-lock-s F     hold the beam on the wake direction (default 4)\n"
               "  --doa-during-playback  also track direction while the speaker plays\n"
               "  --amixer-card DEV   card for set_gain (AC107 'Channel N PGA Gain')\n"
               "  --dump FILE         write AEC output + reference (S16, mics+1 channels)\n"
               "  --no-output         do not write the beam to stdout\n"
               "  --out-cmd CMD       write the beam to CMD's stdin instead (sh -c; restarted 1 s after it exits;\n"
               "                      never blocks: audio is dropped while CMD does not read)\n",
               kVersion);
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep))
    if (!item.empty()) out.push_back(item);
  return out;
}

bool parse_args(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "voicefront: %s needs a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") return false;
    if (a == "--channels") o.channels = std::atoi(next("--channels").c_str());
    else if (a == "--mics") {
      o.mics.clear();
      for (const auto& s : split(next("--mics"), ',')) o.mics.push_back(std::atoi(s.c_str()));
    } else if (a == "--ref") o.ref = std::atoi(next("--ref").c_str());
    else if (a == "--no-aec") o.aec = false;
    else if (a == "--aec-delay-ms") o.aec_delay_ms = std::atoi(next("--aec-delay-ms").c_str());
    else if (a == "--aec-tuning") o.aec_tuning = next("--aec-tuning");
    else if (a == "--pitch-mm") o.pitch_mm = std::strtof(next("--pitch-mm").c_str(), nullptr);
    else if (a == "--beamformer") o.beamformer = next("--beamformer");
    else if (a == "--vad-snr-db") o.vad_snr_db = std::strtof(next("--vad-snr-db").c_str(), nullptr);
    else if (a == "--ns") o.ns = next("--ns");
    else if (a == "--no-agc") o.agc = false;
    else if (a == "--model") o.model_dir = next("--model");
    else if (a == "--keywords") o.keywords = split(next("--keywords"), ';');
    else if (a == "--kw-threshold") o.kw_threshold = std::strtof(next("--kw-threshold").c_str(), nullptr);
    else if (a == "--kw-score") o.kw_score = std::strtof(next("--kw-score").c_str(), nullptr);
    else if (a == "--no-kws") o.kws = false;
    else if (a == "--oww") o.oww = next("--oww");
    else if (a == "--kws-trace") o.kws_trace = true;
    else if (a == "--kws-input") o.kws_input = next("--kws-input");
    else if (a == "--kws-agc") o.kws_agc = std::atoi(next("--kws-agc").c_str()) != 0;
    else if (a == "--kws-ns") o.kws_ns = next("--kws-ns");
    else if (a == "--kws-gain-db") o.kws_gain_db = std::strtof(next("--kws-gain-db").c_str(), nullptr);
    else if (a == "--sync-kws") o.sync_kws = std::atoi(next("--sync-kws").c_str());
    else if (a == "--serial") o.serial = next("--serial");
    else if (a == "--baud") o.baud = std::atoi(next("--baud").c_str());
    else if (a == "--doa-hz") o.doa_hz = std::strtof(next("--doa-hz").c_str(), nullptr);
    else if (a == "--status-s") o.status_s = std::strtof(next("--status-s").c_str(), nullptr);
    else if (a == "--wake-lock-s") o.wake_lock_s = std::strtof(next("--wake-lock-s").c_str(), nullptr);
    else if (a == "--doa-during-playback") o.doa_during_playback = true;
    else if (a == "--amixer-card") o.amixer_card = next("--amixer-card");
    else if (a == "--dump") o.dump = next("--dump");
    else if (a == "--no-output") o.output = false;
    else if (a == "--out-cmd") o.out_cmd = next("--out-cmd");
    else {
      std::fprintf(stderr, "voicefront: unknown option %s\n", a.c_str());
      return false;
    }
  }
  if (o.channels < 1 || o.mics.size() < 2) return false;
  for (int m : o.mics)
    if (m < 0 || m >= o.channels) return false;
  if (o.ref >= o.channels) return false;
  if (o.ref < 0) o.aec = false;
  if (o.kws_input != "beam" && o.kws_input != "mic" && o.kws_input != "linear") return false;
  if (o.kws_input == "linear" && !o.aec) o.kws_input = "mic";
  return true;
}

double now_s(clockid_t c) {
  timespec ts{};
  clock_gettime(c, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

bool read_full(int fd, void* buf, size_t n) {
  auto* p = static_cast<char*>(buf);
  size_t got = 0;
  while (got < n) {
    const ssize_t r = ::read(fd, p + got, n - got);
    if (r > 0) got += static_cast<size_t>(r);
    else if (r < 0 && errno == EINTR) continue;
    else return false;
  }
  return true;
}

bool write_full(int fd, const void* buf, size_t n) {
  const auto* p = static_cast<const char*>(buf);
  size_t off = 0;
  while (off < n) {
    const ssize_t r = ::write(fd, p + off, n - off);
    if (r > 0) off += static_cast<size_t>(r);
    else if (r < 0 && errno == EINTR) continue;
    else return false;
  }
  return true;
}

std::string fmt(const char* f, double v) {
  char b[32];
  std::snprintf(b, sizeof(b), f, v);
  return b;
}

std::string str_list(const std::vector<std::string>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + json_str(v[i]);
  return s + "]";
}

// Keyword spotter, either inline or on its own thread (the audio loop never waits for it).
class KwsRunner {
 public:
  KwsRunner(WakeSpotter* k, bool async) : k_(k), async_(async) {
    if (async_) th_ = std::thread([this] { loop(); });
  }
  ~KwsRunner() { stop(); }

  void push(const float* x, int n) {
    if (!async_) {
      auto h = k_->feed(x, n);
      std::lock_guard<std::mutex> lk(mu_);
      hits_.insert(hits_.end(), h.begin(), h.end());
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    q_.emplace_back(x, x + n);
    if (q_.size() > 300) q_.pop_front();  // 3 s behind: drop rather than grow
    cv_.notify_one();
  }

  std::vector<KwsHit> take() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<KwsHit> h;
    h.swap(hits_);
    return h;
  }

  // Runs f on the spotter's thread (between two feeds) and returns its result.
  template <class F>
  auto call(F f) -> decltype(f()) {
    if (!async_) return f();
    std::unique_lock<std::mutex> lk(mu_);
    jobs_.emplace_back([&] { f_result_ = f(); });
    cv_.notify_one();
    done_cv_.wait(lk, [&] { return jobs_.empty() && !running_job_; });
    return f_result_;
  }

  bool peak_score_known() const { return std::string(k_->engine()) == "oww"; }
  float peak_score() { return k_->peak_score(); }  // atomic in the engine: no need to wait for its thread

  void drain() {
    if (!async_) return;
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return q_.empty() && !busy_; });
  }

  void stop() {
    if (!async_ || !th_.joinable()) return;
    {
      std::lock_guard<std::mutex> lk(mu_);
      quit_ = true;
    }
    cv_.notify_one();
    th_.join();
  }

 private:
  void loop() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cv_.wait(lk, [&] { return quit_ || !q_.empty() || !jobs_.empty(); });
      if (quit_) return;
      if (!jobs_.empty()) {
        auto job = std::move(jobs_.front());
        jobs_.pop_front();
        running_job_ = true;
        lk.unlock();
        job();
        lk.lock();
        running_job_ = false;
        done_cv_.notify_all();
        continue;
      }
      std::vector<float> x = std::move(q_.front());
      q_.pop_front();
      busy_ = true;
      lk.unlock();
      auto h = k_->feed(x.data(), static_cast<int>(x.size()));
      lk.lock();
      busy_ = false;
      hits_.insert(hits_.end(), h.begin(), h.end());
      done_cv_.notify_all();
    }
  }

  WakeSpotter* k_;
  bool async_;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  std::deque<std::vector<float>> q_;
  std::deque<std::function<void()>> jobs_;
  std::vector<KwsHit> hits_;
  bool quit_ = false, busy_ = false, running_job_ = false;
  bool f_result_ = false;
};

// The beam's consumer (e.g. aplay into the USB gadget). aplay exits with EIO whenever the USB host stops
// reading; the wake word and direction must keep working, so the consumer is respawned instead of us dying.
class OutProc {
 public:
  explicit OutProc(std::string cmd) : cmd_(std::move(cmd)) {}
  ~OutProc() { close_pipe(); }
  void write(const int16_t* x, int n, double now) {
    reap();
    if (fd_ < 0) {
      if (now < retry_at_) return;
      spawn();
      if (fd_ < 0) {
        retry_at_ = now + 1.0;
        return;
      }
    }
    const ssize_t r = ::write(fd_, x, static_cast<size_t>(n) * sizeof(int16_t));
    if (r < 0 && errno == EAGAIN) {
      ++dropped_;  // the consumer is not reading (pipe full): drop this chunk
    } else if (r < 0) {
      close_pipe();  // EPIPE: it exited
      retry_at_ = now + 1.0;
      ++restarts_;
    }
  }
  long restarts() const { return restarts_; }

 private:
  void spawn() {
    int p[2];
    if (pipe(p) != 0) return;
    const pid_t pid = fork();
    if (pid == 0) {
      dup2(p[0], 0);
      ::close(p[0]);
      ::close(p[1]);
      execl("/bin/sh", "sh", "-c", cmd_.c_str(), static_cast<char*>(nullptr));
      _exit(127);
    }
    ::close(p[0]);
    if (pid < 0) {
      ::close(p[1]);
      return;
    }
    fd_ = p[1];
    pid_ = pid;
    fcntl(fd_, F_SETFL, fcntl(fd_, F_GETFL) | O_NONBLOCK);
    fcntl(fd_, F_SETFD, FD_CLOEXEC);
  }
  void close_pipe() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }
  void reap() {
    while (waitpid(-1, nullptr, WNOHANG) > 0) {
    }
  }
  std::string cmd_;
  int fd_ = -1;
  pid_t pid_ = -1;
  double retry_at_ = 0.0;
  long dropped_ = 0, restarts_ = 0;
};

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

}  // namespace

int main(int argc, char** argv) {
  Options o;
  if (!parse_args(argc, argv, o)) {
    usage();
    return 2;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGPIPE, SIG_IGN);

  Link link;
  std::string err;
  if (!link.open(o.serial, o.baud, err)) {
    std::fprintf(stderr, "voicefront: %s\n", err.c_str());
    return 1;
  }
  auto log = [&](const std::string& msg) {
    std::fprintf(stderr, "voicefront: %s\n", msg.c_str());
    std::fflush(stderr);
  };

  const int m = static_cast<int>(o.mics.size());
  ArrayConfig ac;
  ac.mics = m;
  ac.pitch_m = o.pitch_mm / 1000.0f;
  ac.beamformer = o.beamformer;
  ac.vad_snr_db = o.vad_snr_db;
  ArrayProcessor array(ac);
  array.set_doa_during_playback(o.doa_during_playback);

  std::unique_ptr<EchoCanceller> aec;
  if (o.aec) aec = std::make_unique<EchoCanceller>(m, o.aec_delay_ms, o.kws && o.kws_input == "linear", o.aec_tuning);
  PostFilter post(o.ns, o.agc);
  const bool kws_filter = o.kws_ns != "off" || o.kws_agc;
  const float kws_gain = std::pow(10.0f, o.kws_gain_db / 20.0f);
  PostFilter kws_post(o.kws_ns, o.kws_agc);  // optional own NS / AGC for the wake-word input
  const int kws_latency = o.kws_input == "beam" ? ArrayProcessor::latency() : 0;

  struct stat st{};
  const bool file_in = fstat(0, &st) == 0 && S_ISREG(st.st_mode);
  const bool sync_kws = o.sync_kws < 0 ? file_in : o.sync_kws != 0;

  std::unique_ptr<WakeSpotter> spotter;
  std::unique_ptr<KwsRunner> kws;
  if (o.kws && !o.oww.empty()) {  // one trained phrase, openWakeWord
    auto s = std::make_unique<OwwSpotter>();
    OwwConfig oc;
    oc.model = o.oww;
    oc.threshold = o.kw_threshold > 0.0f ? o.kw_threshold : 0.5f;
    oc.threads = o.kws_threads;
    oc.trace = o.kws_trace;
    struct stat fs{};
    if (stat((o.model_dir + "/melspectrogram.onnx").c_str(), &fs) == 0) oc.feature_dir = o.model_dir;
    if (!s->init(oc, err)) {
      log("openwakeword: " + err);
      return 1;
    }
    spotter = std::move(s);
  } else if (o.kws) {  // open vocabulary, sherpa-onnx
    auto s = std::make_unique<KeywordSpotter>();
    KwsConfig kc;
    kc.model_dir = o.model_dir;
    kc.keywords = o.keywords;
    kc.threshold = o.kw_threshold > 0.0f ? o.kw_threshold : 0.25f;
    kc.score = o.kw_score;
    kc.threads = o.kws_threads;
    if (!s->init(kc, err)) {
      log("keyword spotter: " + err);
      return 1;
    }
    std::string e2;
    s->set_keywords(o.keywords, kc.threshold, o.kw_score, e2);  // normalises the names
    spotter = std::move(s);
  }
  if (spotter) kws = std::make_unique<KwsRunner>(spotter.get(), !sync_kws);

  FILE* dump = o.dump.empty() ? nullptr : std::fopen(o.dump.c_str(), "wb");
  std::unique_ptr<OutProc> out_proc;
  if (!o.out_cmd.empty()) out_proc = std::make_unique<OutProc>(o.out_cmd);

  auto hello = [&] {
    link.send(std::string("{\"type\":\"hello\",\"name\":\"voicefront\",\"version\":\"") + kVersion +
              "\",\"mics\":" + std::to_string(m) + ",\"rate\":16000,\"aec\":" + (o.aec ? "true" : "false") +
              ",\"beamformer\":" + json_str(array.beamformer()) +
              ",\"keywords\":" + str_list(o.kws ? spotter->keywords() : std::vector<std::string>{}) + "}");
  };
  hello();

  const int c = o.channels;
  std::vector<int16_t> in(static_cast<size_t>(kChunk) * c), mic(static_cast<size_t>(kChunk) * m), ref(kChunk, 0);
  std::vector<std::vector<float>> mf(m, std::vector<float>(kChunk));
  std::vector<const float*> mp(m);
  for (int i = 0; i < m; ++i) mp[i] = mf[i].data();
  std::vector<float> beam, kbuf(kChunk);
  std::vector<int16_t> out(kChunk), kout(kChunk), dbuf(static_cast<size_t>(kChunk) * (m + 1));

  // state
  int64_t chunks = 0;
  float ref_floor_db = -70.0f;
  int far_hang = 0;
  bool far = false, last_speech = false;
  bool fixed = false;
  float fixed_angle = 90.0f, beam_angle = 90.0f;
  double lock_until = -1.0, next_doa = 0.0, next_status = o.status_s;
  double erle_in = 0.0, erle_out = 0.0;
  int erle_n = 0;
  long wakes = 0;
  double cpu0 = now_s(CLOCK_PROCESS_CPUTIME_ID), wall0 = now_s(CLOCK_MONOTONIC);
  double cpu_last = cpu0, wall_last = wall0;
  float cpu_pct = 0.0f;
  const double t_start = wall0;

  auto reply = [&](const Json& cmd, bool ok, const std::string& extra, const std::string& error) {
    std::string s = "{\"type\":\"reply\",\"cmd\":" + json_str(cmd.get_str("cmd")) + ",\"ok\":" + (ok ? "true" : "false");
    if (const Json* id = cmd.get("id")) {
      if (id->type == Json::Num) s += ",\"id\":" + fmt("%.0f", id->num);
      else if (id->type == Json::Str) s += ",\"id\":" + json_str(id->str);
    }
    if (!error.empty()) s += ",\"error\":" + json_str(error);
    link.send(s + extra + "}");
  };

  auto status = [&](double t) {
    std::string s = "{\"type\":\"status\",\"t\":" + fmt("%.2f", t) + ",\"cpu\":" + fmt("%.1f", cpu_pct) +
                    ",\"mode\":" + (fixed ? "\"fixed\"" : "\"auto\"") + ",\"beam\":" + fmt("%.1f", beam_angle) +
                    ",\"beamformer\":" + json_str(array.beamformer()) + ",\"speech\":" +
                    (array.speech() ? "true" : "false") + ",\"playback\":" + (far ? "true" : "false") +
                    ",\"snr_db\":" + fmt("%.1f", array.snr_db()) + ",\"wakes\":" + std::to_string(wakes) +
                    ",\"dropped\":" + std::to_string(link.dropped());
    if (out_proc) s += ",\"out_restarts\":" + std::to_string(out_proc->restarts());
    if (erle_n > 50 && erle_out > 0.0) s += ",\"erle_db\":" + fmt("%.1f", 10.0 * std::log10(erle_in / erle_out));
    if (o.kws) {
      s += ",\"keywords\":" + str_list(spotter->keywords()) + ",\"kw_threshold\":" + fmt("%.3f", spotter->threshold()) +
           ",\"kws_engine\":" + json_str(spotter->engine());
      if (kws->peak_score_known()) s += ",\"kw_peak\":" + fmt("%.3f", kws->peak_score());
    }
    link.send(s + "}");
    erle_in = erle_out = 0.0;
    erle_n = 0;
  };

  auto set_gain = [&](int pga_mic, int pga_ref) {
    std::string cmd;
    auto one = [&](int slot, int v) {
      cmd += "amixer -q -D " + o.amixer_card + " cset name='Channel " + std::to_string(slot + 1) + " PGA Gain' " +
             std::to_string(v) + ";";
    };
    if (pga_mic >= 0)
      for (int s : o.mics) one(s, pga_mic);
    if (pga_ref >= 0 && o.ref >= 0) one(o.ref, pga_ref);
    std::thread([cmd] { (void)!std::system(cmd.c_str()); }).detach();
  };

  auto handle = [&](const std::string& line, double t) {
    Json j;
    if (!Json::parse(line, j) || j.type != Json::Obj) {
      link.send("{\"type\":\"reply\",\"ok\":false,\"error\":\"bad json\"}");
      return;
    }
    const std::string cmd = j.get_str("cmd");
    if (cmd == "ping") {  // reply carries the hello fields; a "hello" event alone means voicefront (re)started
      reply(j, true, std::string(",\"version\":\"") + kVersion + "\",\"keywords\":" +
                         str_list(o.kws ? spotter->keywords() : std::vector<std::string>{}), "");
    } else if (cmd == "status") {
      reply(j, true, "", "");
      status(t);
    } else if (cmd == "set_beam") {
      const std::string mode = j.get_str("mode", "fixed");
      if (mode == "auto") {
        fixed = false;
      } else {
        fixed = true;
        fixed_angle = static_cast<float>(j.get_num("angle", fixed_angle));
        beam_angle = fixed_angle;
        array.steer(beam_angle);
      }
      reply(j, true, ",\"mode\":" + json_str(fixed ? "fixed" : "auto") + ",\"beam\":" + fmt("%.1f", beam_angle), "");
    } else if (cmd == "set_keywords") {
      if (!o.kws) return reply(j, false, "", "keyword spotting is off");
      std::vector<std::string> kw;
      if (const Json* a = j.get("keywords"))
        for (const auto& v : a->arr)
          if (v.type == Json::Str) kw.push_back(v.str);
      if (kw.empty()) kw = spotter->keywords();
      const float th = static_cast<float>(j.get_num("threshold", spotter->threshold()));
      const float sc = static_cast<float>(j.get_num("score", spotter->score()));
      std::string e;
      const bool ok = kws->call([&] { return spotter->set_keywords(kw, th, sc, e); });
      reply(j, ok, ",\"keywords\":" + str_list(spotter->keywords()), ok ? "" : e);
    } else if (cmd == "set") {
      if (j.get("beamformer")) array.set_beamformer(j.get_str("beamformer"));
      if (j.has_num("vad_snr_db")) array.set_vad_snr_db(static_cast<float>(j.get_num("vad_snr_db", 8)));
      if (j.has_num("doa_hz")) o.doa_hz = static_cast<float>(j.get_num("doa_hz", o.doa_hz));
      if (j.has_num("status_s")) o.status_s = static_cast<float>(j.get_num("status_s", o.status_s));
      if (j.has_num("wake_lock_s")) o.wake_lock_s = static_cast<float>(j.get_num("wake_lock_s", o.wake_lock_s));
      if (j.get("doa_during_playback")) array.set_doa_during_playback(j.get_bool("doa_during_playback", false));
      reply(j, true, "", "");
    } else if (cmd == "set_gain") {
      if (o.amixer_card.empty()) return reply(j, false, "", "no --amixer-card");
      const int gm = static_cast<int>(j.get_num("mic", -1)), gr = static_cast<int>(j.get_num("ref", -1));
      if (gm > 31 || gr > 31) return reply(j, false, "", "gain is 0-31");
      set_gain(gm, gr);
      reply(j, true, "", "");
    } else {
      reply(j, false, "", "unknown cmd");
    }
  };

  log(std::string("running: ") + std::to_string(c) + " ch in, mics " + std::to_string(m) +
      (o.aec ? ", aec on ref " + std::to_string(o.ref) : ", no aec") + ", " + array.beamformer() +
      (o.kws ? ", " + std::string(spotter->engine()) + " kws on " + o.kws_input + (sync_kws ? " (inline)" : " (thread)")
             : ", no kws"));

  while (!g_stop && read_full(0, in.data(), in.size() * sizeof(int16_t))) {
    ++chunks;
    const double t = static_cast<double>(chunks) * kChunk / kRate;  // audio clock

    double rs = 0.0, mic0_in = 0.0;
    for (int i = 0; i < kChunk; ++i) {
      const int16_t* f = &in[static_cast<size_t>(i) * c];
      if (o.ref >= 0) ref[i] = f[o.ref];
      rs += static_cast<double>(ref[i]) * ref[i];
      for (int k = 0; k < m; ++k) mic[static_cast<size_t>(i) * m + k] = f[o.mics[k]];
      mic0_in += static_cast<double>(mic[static_cast<size_t>(i) * m]) * mic[static_cast<size_t>(i) * m];
    }

    // speaker activity from the reference level (floor tracker + absolute minimum, 300 ms hangover)
    if (o.ref >= 0) {
      const float rdb = 10.0f * static_cast<float>(std::log10(rs / kChunk / (32768.0 * 32768.0) + 1e-12));
      if (rdb < ref_floor_db) ref_floor_db = 0.9f * ref_floor_db + 0.1f * rdb;
      else ref_floor_db = std::min(ref_floor_db + 0.01f, rdb);
      if (rdb > std::max(ref_floor_db + 12.0f, -62.0f)) far_hang = 30;
      else if (far_hang > 0) --far_hang;
      far = far_hang > 0;
    }

    if (aec) aec->process(ref.data(), mic.data());
    if (kws && o.kws_input != "beam") {
      bool got = false;
      if (o.kws_input == "linear") got = aec->linear(kbuf.data());
      if (!got)
        for (int i = 0; i < kChunk; ++i) kbuf[i] = mic[static_cast<size_t>(i) * m] / 32768.0f;
      if (kws_filter) {
        for (int i = 0; i < kChunk; ++i)
          kout[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, std::round(kbuf[i] * 32768.0f))));
        kws_post.process(kout.data());
        for (int i = 0; i < kChunk; ++i) kbuf[i] = kout[i] / 32768.0f;
      }
      for (int i = 0; i < kChunk; ++i) kbuf[i] = std::max(-1.0f, std::min(1.0f, kbuf[i] * kws_gain));
      kws->push(kbuf.data(), kChunk);
    }
    if (far) {
      double mo = 0.0;
      for (int i = 0; i < kChunk; ++i) mo += static_cast<double>(mic[static_cast<size_t>(i) * m]) * mic[static_cast<size_t>(i) * m];
      erle_in += mic0_in;
      erle_out += mo + 1.0;
      ++erle_n;
    }
    if (dump) {
      for (int i = 0; i < kChunk; ++i) {
        for (int k = 0; k < m; ++k) dbuf[static_cast<size_t>(i) * (m + 1) + k] = mic[static_cast<size_t>(i) * m + k];
        dbuf[static_cast<size_t>(i) * (m + 1) + m] = ref[i];
      }
      std::fwrite(dbuf.data(), sizeof(int16_t), dbuf.size(), dump);
    }

    for (int k = 0; k < m; ++k)
      for (int i = 0; i < kChunk; ++i) mf[k][i] = mic[static_cast<size_t>(i) * m + k] / 32768.0f;
    array.push(mp.data(), kChunk, far, beam);

    // beam steering
    if (fixed) {
      beam_angle = fixed_angle;
    } else if (t >= lock_until && array.doa_fresh() && array.doa_confidence() > 0.3f) {
      beam_angle += 0.2f * (array.doa() - beam_angle);
    }
    array.steer(beam_angle);

    // beam out: post filter, stdout, keyword spotter
    size_t used = 0;
    while (beam.size() - used >= static_cast<size_t>(kChunk)) {
      for (int i = 0; i < kChunk; ++i) {
        const float v = std::round(beam[used + i] * 32768.0f);
        out[i] = static_cast<int16_t>(std::max(-32768.0f, std::min(32767.0f, v)));
      }
      if (kws && o.kws_input == "beam") {  // the beam before noise suppression
        for (int i = 0; i < kChunk; ++i) kout[i] = out[i];
        if (kws_filter) kws_post.process(kout.data());
        for (int i = 0; i < kChunk; ++i) kbuf[i] = std::max(-1.0f, std::min(1.0f, kout[i] / 32768.0f * kws_gain));
        kws->push(kbuf.data(), kChunk);
      }
      used += kChunk;
      post.process(out.data());
      if (out_proc) {
        out_proc->write(out.data(), kChunk, now_s(CLOCK_MONOTONIC));
      } else if (o.output && !write_full(1, out.data(), out.size() * sizeof(int16_t))) {
        log("stdout closed");
        g_stop = true;
        break;
      }
    }
    beam.erase(beam.begin(), beam.begin() + static_cast<long>(used));

    // wake words: direction over the keyword's own frames
    if (kws) {
      for (const auto& h : kws->take()) {
        const int64_t end = h.sample - kws_latency;
        const float dur = h.duration_s > 0.2f ? h.duration_s : 1.0f;
        const int64_t s0 = end - static_cast<int64_t>((dur + 0.6f) * kRate);
        float deg = beam_angle, conf = 0.0f;
        const bool got = array.angle_between(s0, end, deg, conf);
        if (got && !fixed) {
          beam_angle = deg;
          array.steer(beam_angle);
          lock_until = t + o.wake_lock_s;
        }
        ++wakes;
        link.send("{\"type\":\"wake\",\"keyword\":" + json_str(h.keyword) + ",\"angle\":" +
                  (got ? fmt("%.1f", deg) : std::string("null")) + ",\"conf\":" + fmt("%.2f", conf) +
                  ",\"playback\":" + (far ? "true" : "false") + ",\"t\":" + fmt("%.2f", static_cast<double>(end) / kRate) + "}");
      }
    }

    // voice activity / direction events
    if (array.speech() != last_speech) {
      last_speech = array.speech();
      link.send(std::string("{\"type\":\"vad\",\"speech\":") + (last_speech ? "true" : "false") + ",\"t\":" + fmt("%.2f", t) + "}");
    }
    if (o.doa_hz > 0 && array.doa_fresh() && t >= next_doa) {
      next_doa = t + 1.0 / o.doa_hz;
      link.send("{\"type\":\"doa\",\"angle\":" + fmt("%.1f", array.doa()) + ",\"conf\":" + fmt("%.2f", array.doa_confidence()) +
                ",\"beam\":" + fmt("%.1f", beam_angle) + ",\"t\":" + fmt("%.2f", t) + "}");
    }

    // host commands and status (checked every 50 ms)
    if (chunks % 5 == 0) {
      for (const auto& l : link.poll()) handle(l, t);
      if (o.status_s > 0 && t >= next_status) {
        const double cpu = now_s(CLOCK_PROCESS_CPUTIME_ID), wall = now_s(CLOCK_MONOTONIC);
        if (wall > wall_last) cpu_pct = static_cast<float>(100.0 * (cpu - cpu_last) / (wall - wall_last));
        cpu_last = cpu;
        wall_last = wall;
        next_status = t + o.status_s;
        status(t);
      }
    }
  }

  if (kws) {
    kws->drain();
    for (const auto& h : kws->take())
      link.send("{\"type\":\"wake\",\"keyword\":" + json_str(h.keyword) + ",\"angle\":null,\"t\":" +
                fmt("%.2f", static_cast<double>(h.sample - kws_latency) / kRate) +
                (h.score > 0.0f ? ",\"score\":" + fmt("%.3f", h.score) : "") + "}");
    kws->stop();
  }
  if (dump) std::fclose(dump);
  const double wall = now_s(CLOCK_MONOTONIC) - t_start, cpu = now_s(CLOCK_PROCESS_CPUTIME_ID) - cpu0;
  const double audio = static_cast<double>(chunks) * kChunk / kRate;
  std::fprintf(stderr, "voicefront: %.1f s audio in %.1f s, cpu %.2f s (%.1f %% of real time), %ld wakes\n", audio, wall,
               cpu, audio > 0 ? 100.0 * cpu / audio : 0.0, wakes);
  return 0;
}
