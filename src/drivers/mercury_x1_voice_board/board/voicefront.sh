#!/bin/sh
# Runs ON the voice board instead of the stock demo (installed by scripts/install_voicefront.sh, started from
# /etc/rc.local). Our front end, see ../voicefront/README.md:
#   AC107 mics + speaker echo reference (Ac107Snoop) -> voicefront -> processed beam on USB channel 5
#   wake word, direction, voice activity, status: JSON lines on UART2 (/dev/ttyS2 -> CH343 -> Orin /dev/xf_mic)
# Settings: /data/build/voicefront/voicefront.conf (shell variables, see voicefront.conf.default).
trap "" HUP
D=/data/build/voicefront
C=hw:sndac10710036
while true; do
    MIC_GAIN=4      # AC107 PGA (0..31): the robot's own voice at full speaker volume must not clip the mics
    REF_GAIN=20     # PGA of the speaker echo reference (TDM slot 6 = "Channel 7")
    KWS_GAIN_DB=20  # the wake-word input is boosted: at PGA 4 a talker 2 m away peaks around -33 dBFS
    WAKE_MODEL=""   # openWakeWord model file; empty = the trained hey_mercury.onnx if present, else sherpa KWS
    KW_THRESHOLD="" # empty = the engine's default (0.5 openWakeWord, 0.25 sherpa)
    VF_ARGS=""
    [ -f $D/voicefront.conf ] && . $D/voicefront.conf   # re-read on every restart (killall voicefront)
    for n in 1 2 3 4; do amixer -q -D $C cset name="Channel $n PGA Gain" "$MIC_GAIN"; done
    amixer -q -D $C cset name="Channel 7 PGA Gain" "$REF_GAIN"
    ENGINE=""
    [ -z "$WAKE_MODEL" ] && [ -f $D/model/hey_mercury.onnx ] && WAKE_MODEL=$D/model/hey_mercury.onnx
    [ -n "$WAKE_MODEL" ] && [ -f "$WAKE_MODEL" ] && ENGINE="--oww $WAKE_MODEL"
    [ -n "$KW_THRESHOLD" ] && ENGINE="$ENGINE --kw-threshold $KW_THRESHOLD"
    # shellcheck disable=SC2086
    arecord -q -D Ac107Snoop -c 8 -r 16000 -f S16_LE -t raw 2>/dev/null \
        | $D/voicefront --channels 8 --mics 0,1,2,3 --ref 6 --model $D/model --serial /dev/ttyS2 \
            --amixer-card $C --out-cmd "aplay -q -D PlaybackDmix -c 1 -r 16000 -f S16_LE -t raw 2>/dev/null" \
            --kws-gain-db "$KWS_GAIN_DB" $ENGINE $VF_ARGS > /dev/null 2> /tmp/voicefront.log
    sleep 1
done
