cmake_minimum_required(VERSION 3.16)
project(voicefront CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies prepared by build_deps.sh (aarch64): WebRTC audio processing + abseil, sherpa-onnx static libs.
set(VF_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/deps/prefix" CACHE PATH "prefix made by build_deps.sh")
# WebRTC source tree (apm.cpp creates AEC3 itself and needs its internal header)
set(VF_WEBRTC_SRC "${CMAKE_CURRENT_SOURCE_DIR}/deps/webrtc-audio-processing/webrtc" CACHE PATH "WebRTC source")
# Fully static binary: runs on the voice board's old glibc (Tina Linux, glibc 2.23) as well as on the Orin.
option(VF_STATIC "link fully static" ON)

add_executable(voicefront src/main.cpp src/array.cpp src/apm.cpp src/kws.cpp src/link.cpp)
target_compile_options(voicefront PRIVATE -O2 -fcx-limited-range -Wall -Wextra -Wno-unused-parameter)
target_include_directories(voicefront PRIVATE
  ${VF_DEPS}/include/webrtc-audio-processing-2
  ${VF_DEPS}/include/absl
  ${VF_DEPS}/include)
set_source_files_properties(src/apm.cpp PROPERTIES
  COMPILE_DEFINITIONS "WEBRTC_POSIX;WEBRTC_LINUX;WEBRTC_ARCH_ARM64;WEBRTC_APM_DEBUG_DUMP=0;WEBRTC_LIBRARY_IMPL;NDEBUG"
  INCLUDE_DIRECTORIES "${VF_WEBRTC_SRC}")

set(SHERPA_LIBS
  sherpa-onnx-c-api sherpa-onnx-core kaldi-decoder-core sherpa-onnx-kaldifst-core sherpa-onnx-fstfar
  sherpa-onnx-fst kaldi-native-fbank-core kissfft-float ssentencepiece_core utf8proc onnxruntime)
file(GLOB ABSL_LIBS ${VF_DEPS}/lib/libabsl_*.a)
target_link_directories(voicefront PRIVATE ${VF_DEPS}/lib)
target_link_libraries(voicefront PRIVATE
  -Wl,--start-group ${SHERPA_LIBS} webrtc-audio-processing-2 ${ABSL_LIBS} -Wl,--end-group
  pthread dl m rt)
if(VF_STATIC)
  target_link_options(voicefront PRIVATE -static)
endif()
