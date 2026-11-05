#!/bin/bash
# Set up the robot's own language model (JetPack 7.2 / CUDA 13.2, Orin sm_87): build llama.cpp's llama-server,
# download the 4-bit models, install the systemd unit mercury-llm (127.0.0.1:8091, see run.sh).
#   server/robot_llm/setup.sh            build in ~/llm/llama.cpp, models in ~/llm/models, install + start
#   server/robot_llm/setup.sh --remove   stop + remove the unit (build and models stay)
# Pick the model in ~/.config/mercury_x1/llm.env (LLM_MODEL=...), then: sudo systemctl restart mercury-llm
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
if [ "$1" = "--remove" ]; then
  sudo systemctl disable --now mercury-llm 2>/dev/null || true
  sudo rm -f /etc/systemd/system/mercury-llm.service
  sudo systemctl daemon-reload
  exit 0
fi
D=~/llm
mkdir -p "$D/models"
HF=https://huggingface.co
[ -s "$D/models/Qwen3.5-4B-Q4_K_M.gguf" ] || curl -fL -C - -o "$D/models/Qwen3.5-4B-Q4_K_M.gguf" \
  "$HF/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
[ -s "$D/models/Nemotron3-Nano-4B-Q4_K_M.gguf" ] || curl -fL -C - -o "$D/models/Nemotron3-Nano-4B-Q4_K_M.gguf" \
  "$HF/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/main/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"
[ -d "$D/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$D/llama.cpp"
export PATH=/usr/local/cuda/bin:$PATH
cmake -S "$D/llama.cpp" -B "$D/llama.cpp/build" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
# low priority, 4 jobs: the robot stack keeps running during the ~35 min build
nice -n 15 cmake --build "$D/llama.cpp/build" -j4 --target llama-server llama-bench
sudo install -m 644 "$REPO/config/systemd/mercury-llm.service" /etc/systemd/system/mercury-llm.service
sudo systemctl daemon-reload
sudo systemctl restart mercury-llm
echo "mercury-llm started (127.0.0.1:8091); model load ~20 s"
