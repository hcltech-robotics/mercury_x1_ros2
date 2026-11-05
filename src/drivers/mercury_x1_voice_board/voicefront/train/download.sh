#!/bin/bash
# Download TTS checkpoint, precomputed negative features, MIT RIRs
set -x
cd ~/oww_hey_mercury
wget -q -c -O piper-sample-generator/models/en_US-libritts_r-medium.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt' &
wget -q -c 'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy' &
wget -q -c 'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy' &
mkdir -p mit_rirs && cd mit_rirs
curl -s "https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/main/16khz" | python3 -c 'import sys,json; [print(x["path"]) for x in json.load(sys.stdin)]' > ../rir_list.txt
sed 's#^#https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses/resolve/main/#' ../rir_list.txt | xargs -P 8 -n 1 wget -q -c
cd ..; ls mit_rirs | wc -l
wait
ls -la *.npy piper-sample-generator/models/*.pt
echo DL_DONE
