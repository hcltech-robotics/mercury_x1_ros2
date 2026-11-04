#!/usr/bin/env bash
# Fetches and builds the voicefront dependencies on an aarch64 Linux host (the Orin), then builds voicefront.
#   ./build_deps.sh            # deps (once) + build -> build/voicefront, model/ filled
# Needs: git, curl, cmake, meson, ninja, g++ (apt: meson ninja-build cmake g++).
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
DEPS=$HERE/deps
PREFIX=$DEPS/prefix
WAP_TAG=v2.1
SHERPA_VER=2.0.1
KWS=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
JOBS=${JOBS:-3}   # keep the Orin's 12 V converter happy (see the JetPack 7.2 devlog)

[ "$(uname -m)" = aarch64 ] || { echo "run this on aarch64 (the Orin)"; exit 1; }
mkdir -p "$DEPS" "$PREFIX/lib" "$PREFIX/include"

# WebRTC audio processing (AEC3, NS, AGC2) + abseil, static
if [ ! -f "$PREFIX/lib/libwebrtc-audio-processing-2.a" ]; then
  [ -d "$DEPS/webrtc-audio-processing" ] ||
    git clone -q https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing.git "$DEPS/webrtc-audio-processing"
  cd "$DEPS/webrtc-audio-processing"
  git checkout -q $WAP_TAG
  [ -d build ] || meson setup build --prefix="$DEPS/wap" -Ddefault_library=static --buildtype=release
  nice ninja -C build -j"$JOBS"
  ninja -C build install >/dev/null
  cp "$DEPS"/wap/lib/*/libwebrtc-audio-processing-2.a "$PREFIX/lib/"
  cp -r "$DEPS"/wap/include/webrtc-audio-processing-2 "$PREFIX/include/"
  ABSL=$(ls -d build/subprojects/abseil-cpp-*)
  cp "$ABSL"/libabsl_*.a "$PREFIX/lib/"
  mkdir -p "$PREFIX/include/absl"
  cp -r "subprojects/$(basename "$ABSL")/absl" "$PREFIX/include/absl/"
  cd "$HERE"
fi

# sherpa-onnx static libraries (incl. onnxruntime) + C API header
if [ ! -f "$PREFIX/lib/libsherpa-onnx-c-api.a" ]; then
  R=https://github.com/k2-fsa/sherpa-onnx/releases/download
  curl -fsSL -o "$DEPS/so.tar.bz2" "$R/v$SHERPA_VER/sherpa-onnx-v$SHERPA_VER-linux-aarch64-static-lib.tar.bz2"
  tar xjf "$DEPS/so.tar.bz2" -C "$DEPS"
  cp "$DEPS/sherpa-onnx-v$SHERPA_VER-linux-aarch64-static-lib"/lib/*.a "$PREFIX/lib/"
  mkdir -p "$PREFIX/include/sherpa-onnx/c-api"
  curl -fsSL -o "$PREFIX/include/sherpa-onnx/c-api/c-api.h" \
    "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v$SHERPA_VER/sherpa-onnx/c-api/c-api.h"
fi

# English keyword model (Apache-2.0); bpe.vocab is in the repo (exported from the model's bpe.model)
if [ ! -f model/encoder.int8.onnx ]; then
  curl -fsSL -o "$DEPS/kws.tar.bz2" "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$KWS.tar.bz2"
  tar xjf "$DEPS/kws.tar.bz2" -C "$DEPS"
  cp "$DEPS/$KWS/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx" model/encoder.int8.onnx
  cp "$DEPS/$KWS/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx" model/decoder.int8.onnx
  cp "$DEPS/$KWS/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx" model/joiner.int8.onnx
  cp "$DEPS/$KWS/tokens.txt" model/
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
nice cmake --build build -j"$JOBS"
file build/voicefront
