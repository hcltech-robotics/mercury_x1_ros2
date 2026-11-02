#!/bin/bash
# Set up Kokoro speech on the robot's GPU (JetPack 6.2 / CUDA 12.6, Python 3.10), then install + start the unit.
#   server/robot_tts/setup.sh            venv ~/mercury-tts/.venv, models ~/mercury-tts/models, systemd mercury-tts
#   server/robot_tts/setup.sh --remove   stop + remove the unit (venv and models stay)
# Logs: journalctl -u mercury-tts
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
if [ "$1" = "--remove" ]; then
  sudo systemctl disable --now mercury-tts 2>/dev/null || true
  sudo rm -f /etc/systemd/system/mercury-tts.service
  sudo systemctl daemon-reload
  exit 0
fi
D=~/mercury-tts
mkdir -p "$D/models"
[ -x "$D/.venv/bin/python" ] || python3 -m venv "$D/.venv"
PIP="$D/.venv/bin/pip"
"$PIP" install -q --upgrade pip
# the Jetson AI Lab onnxruntime-gpu wheel (CUDA/TensorRT providers) is built against NumPy 1.x; kokoro-onnx
# pins NumPy 2 but runs fine on 1.26, hence --no-deps
"$PIP" install -q --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple onnxruntime-gpu==1.23.0
"$PIP" install -q --no-deps kokoro-onnx==0.6.1
"$PIP" install -q "numpy<2" phonemizer-fork espeakng-loader colorlog aiohttp protobuf sympy flatbuffers coloredlogs packaging
REL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -s "$D/models/$f" ] || curl -L --fail -o "$D/models/$f" "$REL/$f"
done
"$D/.venv/bin/python" -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers()"
sudo install -m 644 "$REPO/config/systemd/mercury-tts.service" /etc/systemd/system/mercury-tts.service
sudo systemctl daemon-reload
sudo systemctl enable --now mercury-tts
sudo systemctl restart mercury-tts
echo "mercury-tts started (127.0.0.1:8097); first start ~15 s (warm-up)"
