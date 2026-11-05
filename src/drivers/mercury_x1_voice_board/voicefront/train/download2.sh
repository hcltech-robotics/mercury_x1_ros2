#!/bin/bash
# Download AudioSet balanced-train parquet shards + FMA small zip (background noise sources)
set -x
cd ~/oww_hey_mercury
mkdir -p audioset fma_zip
for i in 00 01 02; do wget -q -c -O audioset/bal_train_$i.parquet "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train/$i.parquet" & done
wget -q -c -O fma_zip/fma_small.zip "https://os.unil.cloud.switch.ch/fma/fma_small.zip" &
wait
ls -la audioset fma_zip
echo DL2_DONE
