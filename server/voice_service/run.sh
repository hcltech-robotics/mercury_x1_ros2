#!/bin/bash
# start the speech service from its venv; CUDA / cuDNN for Whisper come from the venv's nvidia-* wheels
cd "$(dirname "$0")"
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec .venv/bin/python -u voice_service.py
