#!/bin/bash
# Multi-channel USB audio from the head voice board (run on the robot).
#   install (default): board gets the 6-channel UAC1 gadget (4 raw mics, echo reference, processed beam) via its
#                      /etc/rc.local + /etc/asound.conf + /data/build/rawmic.sh, applied live; the Orin gets the
#                      ALSA names mx1_array / mx1_mics / mx1_beam and voice chat moves to mx1_beam.
#   revert:            board back to its stock files (mono gadget), Orin voice chat back to plughw:CARD=XFMDPV0018.
# The board is reached once over its USB ADB interface to copy the files; the audio itself is plain USB audio.
set -e
MODE=${1:-install}
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/drivers/mercury_x1_voice_board"
BRD=$PKG/board
ADB="sudo python3 $BRD/adb_tool.py"
dev() { echo /dev/bus/usb/$(lsusb -d 2207:0001 | awk '{print $2"/"$4}' | tr -d :); }
D=$(dev); [ "$D" != /dev/bus/usb/ ] || { echo "voice board (2207:0001) not on USB"; exit 1; }
VENV=$HOME/.config/mercury_x1/voice.env

# Keep the stock files once (on the board's overlay).
$ADB $D shell "[ -f /etc/asound.conf.stock ] || cp -p /etc/asound.conf /etc/asound.conf.stock; [ -f /etc/rc.local.stock ] || cp -p /etc/rc.local /etc/rc.local.stock; echo stock kept"
if [ "$MODE" = install ]; then
    $ADB $D push /etc/asound.conf,644 < $BRD/asound.conf
    $ADB $D push /etc/rc.local,755 < $BRD/rc.local
    $ADB $D push /data/build/rawmic.sh,755 < $BRD/rawmic.sh
elif [ "$MODE" = revert ]; then
    # also undoes scripts/install_voicefront.sh (its beam needs the 6-channel gadget): the demo comes back
    $ADB $D shell "cp -p /etc/asound.conf.stock /etc/asound.conf; cp -p /etc/rc.local.stock /etc/rc.local; rm -f /data/build/rawmic.sh /data/build/voicefront/voicefront.sh; [ -f /data/build/wakeupCron.sh.stock ] && cp -p /data/build/wakeupCron.sh.stock /data/build/wakeupCron.sh; killall voicefront.sh voicefront 2>/dev/null; echo stock restored"
else
    echo "usage: $0 [install|revert]"; exit 2
fi
$ADB $D push /tmp/mx1_apply_live.sh,755 < $BRD/apply_live.sh
$ADB $D shell "sync; start-stop-daemon -S -b -x /tmp/mx1_apply_live.sh; echo applying"

echo "waiting for the board to come back..."
sleep 8
for i in $(seq 1 20); do arecord -l 2>/dev/null | grep -q XFMDPV0018 && break; sleep 1; done
sleep 2
$ADB $(dev) shell "cat /tmp/mx1_apply.log"
grep -E "Channels" /proc/asound/XFMDPV0018/stream0 || true

if [ "$MODE" = install ]; then
    sudo install -m 0644 $PKG/alsa/60-mercury-voice-board.conf /etc/alsa/conf.d/60-mercury-voice-board.conf
    echo snd-aloop | sudo tee /etc/modules-load.d/mercury-aloop.conf >/dev/null; sudo modprobe snd-aloop   # mx1_clean
    MIC=mx1_beam
else
    sudo rm -f /etc/alsa/conf.d/60-mercury-voice-board.conf /etc/modules-load.d/mercury-aloop.conf
    MIC=plughw:CARD=XFMDPV0018,DEV=0
fi
if [ -f "$VENV" ]; then
    if grep -q '^VOICE_MIC=' "$VENV"; then sed -i "s|^VOICE_MIC=.*|VOICE_MIC=$MIC|" "$VENV"; else echo "VOICE_MIC=$MIC" >> "$VENV"; fi
    echo "voice chat mic: $MIC (restart the stack or toggle voice chat to pick it up)"
fi
