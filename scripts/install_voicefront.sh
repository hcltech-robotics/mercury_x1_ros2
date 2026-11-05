#!/bin/bash
# voicefront (our mic-array front end) on the head voice board, replacing the stock iFlytek demo. Run on the robot.
#   install (default): build voicefront on the Orin if needed (static aarch64, runs on the board as is), copy it,
#                      the keyword model and the runner to /data/build/voicefront, disable the demo watchdog,
#                      start it now and at every boot (/etc/rc.local).
#   revert:            stop voicefront and re-enable the demo (its watchdog starts it within seconds).
# Needs the 6-channel USB audio setup (scripts/install_voice_board_audio.sh): the beam goes to USB channel 5.
# The board is reached over its USB ADB interface only to copy files and start/stop processes.
set -e
MODE=${1:-install}
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/drivers/mercury_x1_voice_board"
BRD=$PKG/board
VF=$PKG/voicefront
ADB="sudo python3 $BRD/adb_tool.py"
dev() { echo /dev/bus/usb/$(lsusb -d 2207:0001 | awk '{print $2"/"$4}' | tr -d :); }
D=$(dev); [ "$D" != /dev/bus/usb/ ] || { echo "voice board (2207:0001) not on USB"; exit 1; }
B=/data/build

if [ "$MODE" = install ]; then
    [ -x "$VF/build/voicefront" ] && [ -f "$VF/model/encoder.int8.onnx" ] || "$VF/build_deps.sh"
    strip -o /tmp/voicefront.stripped "$VF/build/voicefront"
    $ADB $D shell "mkdir -p $B/voicefront/model; [ -f $B/wakeupCron.sh.stock ] || cp -p $B/wakeupCron.sh $B/wakeupCron.sh.stock; [ -f /etc/rc.local.stock ] || cp -p /etc/rc.local /etc/rc.local.stock; echo stock kept"
    $ADB $D push $B/voicefront/voicefront.new,755 < /tmp/voicefront.stripped
    for f in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt bpe.vocab; do
        $ADB $D push $B/voicefront/model/$f < "$VF/model/$f"
    done
    for f in melspectrogram.onnx embedding_model.onnx hey_mercury.onnx; do   # openWakeWord engine (if trained)
        [ -f "$VF/model/$f" ] && $ADB $D push $B/voicefront/model/$f < "$VF/model/$f"
    done
    $ADB $D push $B/voicefront/voicefront.sh,755 < "$BRD/voicefront.sh"
    $ADB $D push $B/voicefront/voicefront.conf.default < "$BRD/voicefront.conf.default"
    $ADB $D push /etc/rc.local,755 < "$BRD/rc.local"
    $ADB $D push $B/wakeupCron.sh,755 < "$BRD/wakeupCron.disabled.sh"
    $ADB $D shell "cd $B/voicefront; [ -f voicefront.conf ] || cp voicefront.conf.default voicefront.conf; killall voicefront.sh 2>/dev/null; killall voicefront demo 2>/dev/null; sleep 1; mv voicefront.new voicefront; sync; start-stop-daemon -S -b -x $B/voicefront/voicefront.sh; echo started"
    sleep 6
    $ADB $D shell "ps | grep -E 'voicefront|demo' | grep -v grep; tail -3 /tmp/voicefront.log"
elif [ "$MODE" = revert ]; then
    $ADB $D shell "[ -f $B/wakeupCron.sh.stock ] && cp -p $B/wakeupCron.sh.stock $B/wakeupCron.sh; rm -f $B/voicefront/voicefront.sh; killall voicefront.sh 2>/dev/null; killall voicefront 2>/dev/null; sync; echo 'reverted: the watchdog restarts demo'"
    sleep 8
    $ADB $D shell "ps | grep -E 'voicefront|demo' | grep -v grep"
else
    echo "usage: $0 [install|revert]"; exit 2
fi
