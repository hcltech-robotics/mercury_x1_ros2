#!/bin/bash
# Build the Python 3.11 venv for openWakeWord training (uv)
set -x
export PATH=$HOME/.local/bin:$PATH
cd ~/oww_hey_mercury
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio 2>&1 | tail -3
uv pip install -e ./piper-sample-generator 2>&1 | tail -3
uv pip install -e ./openWakeWord 2>&1 | tail -3
uv pip install mutagen torchinfo torchmetrics speechbrain "audiomentations==0.33.0" torch-audiomentations acoustics pronouncing "datasets<3" deep-phonemizer pyyaml onnx huggingface_hub soundfile 2>&1 | tail -5
python -c "import torch, openwakeword, piper_sample_generator, speechbrain, torch_audiomentations, acoustics, pronouncing; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo ENV_DONE
