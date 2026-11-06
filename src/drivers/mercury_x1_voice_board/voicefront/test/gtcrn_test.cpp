// Offline driver for the native GTCRN: runs model/gtcrn.bin over a file of input frames and writes the output
// frames + the final recurrent state, so test/gtcrn_check.py can compare them with onnxruntime.
//   gtcrn_test gtcrn.bin frames_in.f32 frames_out.f32 [repeat]
// frames_in: float32 [N][257][2] (already scaled by the net's input gain); frames_out: float32 [N][257][2]
// followed by conv_cache [2*16*16*33], tra_cache [2*3*16], inter_cache [2*33*16].
// x86 build: g++ -O3 -ffast-math -std=c++17 -Isrc test/gtcrn_test.cpp src/gtcrn_native.cpp -o gtcrn_test
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gtcrn_native.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: gtcrn_test gtcrn.bin frames_in.f32 frames_out.f32 [repeat]\n");
    return 2;
  }
  GtcrnNative net;
  std::string err;
  if (!net.load(argv[1], 1.0f, err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  FILE* f = std::fopen(argv[2], "rb");
  if (!f) return 1;
  std::vector<float> in;
  float buf[514];
  while (std::fread(buf, sizeof(float), 514, f) == 514) in.insert(in.end(), buf, buf + 514);
  std::fclose(f);
  const size_t n = in.size() / 514;
  std::vector<float> out(in.size());
  const int repeat = argc > 4 ? std::atoi(argv[4]) : 1;
  double best = 1e9;
  for (int r = 0; r < repeat; ++r) {
    net.reset();
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i) net.step(&in[i * 514], &out[i * 514]);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
    if (ms < best) best = ms;
  }
  std::printf("%zu frames, %.4f ms/frame (best of %d)\n", n, best, repeat);
  f = std::fopen(argv[3], "wb");
  if (!f) return 1;
  std::fwrite(out.data(), sizeof(float), out.size(), f);
  std::fwrite(net.conv_cache(), sizeof(float), 2 * 16 * 16 * 33, f);
  std::fwrite(net.tra_cache(), sizeof(float), 2 * 3 * 16, f);
  std::fwrite(net.inter_cache(), sizeof(float), 2 * 33 * 16, f);
  std::fclose(f);
  return 0;
}
