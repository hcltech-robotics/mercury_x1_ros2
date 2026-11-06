#!/bin/bash
# Runs ON THE ORIN (mercury-voicefront.service): the voicefront front end on the raw USB channels of the head voice
# board (4 mics + speaker echo reference, mx1_mics), with GTCRN neural noise suppression, into the ALSA loopback
# that mx1_clean reads. The board's own voicefront keeps the wake word; its beam (mx1_beam) stays as a fallback.
# GTCRN is too slow for the board's Cortex-A53 (1.6x real time, measured 2026-11-05); here it is ~16 % of a core.
# The raw channels reach the Orin with a fixed ~0.6 s USB lag; the board's beam path adds a lag that grows across
# board restarts, which this avoids.
# AEC tuning "call" (WebRTC's default), not the board's "asr": with GTCRN after it, the robot's own voice ends up
# 19 dB lower (-42 vs -23 dBFS after AGC, 2026-11-05) and the person 2 dB lower; neither tuning keeps a person who
# talks over the robot (the recognizer gets nothing in double talk either way).
VF=/home/elephant/mercury_x1_ros2/src/drivers/mercury_x1_voice_board/voicefront
exec arecord -q -D mx1_mics -c 5 -r 16000 -f S16_LE -t raw \
    | "$VF/build/voicefront" --channels 5 --mics 0,1,2,3 --ref 4 --no-kws --no-events --model "$VF/model" --ns gtcrn --aec-tuning call --no-agc --out-gain-db 20 \
          --out-cmd "aplay -q -D mx1_clean_out -c 1 -r 16000 -f S16_LE -t raw"
