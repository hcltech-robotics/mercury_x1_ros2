#!/bin/bash
# copy the speech service to 192.168.11.51 and (re)start it. First install (once, on the server):
#   cd ~/mercury-voice && uv venv -p 3.12 .venv && VIRTUAL_ENV=.venv uv pip install faster-whisper kokoro-onnx \
#     webrtcvad-wheels aiohttp numpy nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
#   models/kokoro-v1.0.onnx + voices-v1.0.bin from github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0)
set -e
HOST="${VOICE_HOST:-hcltech@192.168.11.51}"
cd "$(dirname "$0")"
scp -q voice_service.py run.sh mercury-voice.service "$HOST":mercury-voice/
ssh "$HOST" 'chmod +x ~/mercury-voice/run.sh && sudo -n cp ~/mercury-voice/mercury-voice.service /etc/systemd/system/ &&
  sudo -n systemctl daemon-reload && sudo -n systemctl enable -q mercury-voice && sudo -n systemctl restart mercury-voice'
echo "deployed; logs: ssh $HOST journalctl -u mercury-voice -f"
