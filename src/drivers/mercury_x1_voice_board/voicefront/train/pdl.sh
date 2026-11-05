#!/bin/bash
# Parallel chunked, resumable download of the 17.28 GB ACAV100M feature file from HF (per-connection capped CDN)
cd ~/oww_hey_mercury
URL="https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
SIZE=17280000128; CH=100000000; N=$(( (SIZE + CH - 1) / CH ))
mkdir -p acav_parts
dl() { i=$1; o=$((i*CH)); e=$((o+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1)); f=acav_parts/p$(printf %04d $i)
  for try in 1 2 3 4 5 6 7 8; do
    [ -f $f ] && [ $(stat -c%s $f) -eq $((e-o+1)) ] && return 0
    curl -sL -r $o-$e -o $f "$URL" --max-time 900 && [ $(stat -c%s $f) -eq $((e-o+1)) ] && return 0
    sleep 2
  done; echo "FAILED chunk $i"; }
export -f dl; export URL CH SIZE
seq 0 $((N-1)) | xargs -P 24 -I{} bash -c 'dl {}'
ok=$(ls acav_parts | wc -l); echo "chunks: $ok / $N"
if [ $ok -eq $N ]; then cat acav_parts/p* > openwakeword_features_ACAV100M_2000_hrs_16bit.npy && [ $(stat -c%s openwakeword_features_ACAV100M_2000_hrs_16bit.npy) -eq $SIZE ] && rm -rf acav_parts && echo ACAV_DONE; else echo ACAV_INCOMPLETE; fi
