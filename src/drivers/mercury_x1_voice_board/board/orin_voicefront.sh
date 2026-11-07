#!/bin/bash
# FALLBACK since 2026-11-06 (mercury-voicefront.service, disabled): the same chain the board now runs itself, on the
# Orin, on the raw USB channels of the head voice board (4 mics + speaker echo reference, mx1_mics), into the ALSA
# loopback that mx1_clean reads. Use it (systemctl enable --now mercury-voicefront; VOICE_MIC=mx1_clean in
# ~/.config/mercury_x1/voice.env) if the board cannot keep up. Measured 2026-11-06, board beam vs this: silence
# -61.6/-62.6, person -43.7/-44.8, robot residual -65.0/-66.6 dBFS, same transcripts; the beam arrives 0.6 s
# earlier (the board's output pipe is bounded to 128 ms now; it used to hold 1.5..2.3 s).
# AEC tuning "call" (WebRTC's default), not the board's "asr": with GTCRN after it, the robot's own voice ends up
# 19 dB lower (-42 vs -23 dBFS after AGC, 2026-11-05) and the person 2 dB lower; neither tuning keeps a person who
# talks over the robot (the recognizer gets nothing in double talk either way).
VF=/home/elephant/mercury_x1_ros2/src/drivers/mercury_x1_voice_board/voicefront
exec arecord -q -D mx1_mics -c 5 -r 16000 -f S16_LE -t raw \
    | "$VF/build/voicefront" --channels 5 --mics 0,1,2,3 --ref 4 --no-kws --no-events --model "$VF/model" --ns gtcrn --aec-tuning call --no-agc --out-gain-db 20 \
          --out-cmd "aplay -q -D mx1_clean_out -c 1 -r 16000 -f S16_LE -t raw"
