#!/bin/bash
# Set up speech on the robot (JetPack 6.2 / CUDA 12.6, Python 3.10): Parakeet STT (CPU) + Kokoro TTS (GPU) +
# Silero VAD, then install + start the systemd unit mercury-speech (127.0.0.1:8097).
#   server/robot_speech/setup.sh            venv ~/mercury-speech/.venv, models ~/mercury-speech/models
#   server/robot_speech/setup.sh --remove   stop + remove the unit (venv and models stay)
# Logs: journalctl -u mercury-speech
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
if [ "$1" = "--remove" ]; then
  sudo systemctl disable --now mercury-speech 2>/dev/null || true
  sudo rm -f /etc/systemd/system/mercury-speech.service
  sudo systemctl daemon-reload
  exit 0
fi
D=~/mercury-speech
mkdir -p "$D/models"
[ -x "$D/.venv/bin/python" ] || python3 -m venv "$D/.venv"
PIP="$D/.venv/bin/python -m pip"
$PIP install -q --upgrade pip
# the Jetson AI Lab onnxruntime-gpu wheel (CUDA/TensorRT providers) is built against NumPy 1.x; kokoro-onnx
# pins NumPy 2 but runs fine on 1.26, hence --no-deps
$PIP install -q --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple onnxruntime-gpu==1.23.0
$PIP install -q --no-deps kokoro-onnx==0.6.1 onnx-asr==0.12.0
$PIP install -q "numpy<2" phonemizer-fork espeakng-loader colorlog aiohttp protobuf sympy flatbuffers coloredlogs \
  packaging typing-extensions huggingface_hub
REL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -s "$D/models/$f" ] || curl -L --fail -o "$D/models/$f" "$REL/$f"
done
[ -s "$D/models/silero_vad_v6.onnx" ] || curl -L --fail -o "$D/models/silero_vad_v6.onnx" \
  https://github.com/SYSTRAN/faster-whisper/raw/master/faster_whisper/assets/silero_vad_v6.onnx
[ -s "$D/models/parakeet-tdt-0.6b-v2/encoder-model.int8.onnx" ] || "$D/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('istupakov/parakeet-tdt-0.6b-v2-onnx', local_dir='$D/models/parakeet-tdt-0.6b-v2',
                  allow_patterns=['*int8.onnx', 'config.json', 'vocab.txt'])"
"$D/.venv/bin/python" -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers()"
sudo install -m 644 "$REPO/config/systemd/mercury-speech.service" /etc/systemd/system/mercury-speech.service
sudo systemctl daemon-reload
sudo systemctl enable --now mercury-speech
sudo systemctl restart mercury-speech
echo "mercury-speech started (127.0.0.1:8097); first start ~20 s (warm-up)"
