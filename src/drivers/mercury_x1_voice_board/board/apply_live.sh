#!/bin/sh
# Runs ON the voice board (pushed by scripts/install_voice_board_audio.sh). Re-creates the USB audio gadget with
# the channel mask now in /etc/rc.local and (re)starts rawmic.sh, without a reboot. Log: /tmp/mx1_apply.log
exec > /tmp/mx1_apply.log 2>&1
trap "" HUP
sleep 2                                    # let the ADB session that started us close
G=/sys/kernel/config/usb_gadget/g1
F=$G/functions/uac1.usb0
MASK=$(sed -n 's/^[[:space:]]*echo \(0x[0-9a-f]*\) > .*uac1.usb0\/p_chmask$/\1/p' /etc/rc.local)
[ -n "$MASK" ] || MASK=1
WATCH=$(pidof watch)
[ -n "$WATCH" ] && kill -STOP $WATCH       # the demo watchdog would restart demo mid-way
killall voicefront.sh rawmic.sh 2>/dev/null
killall voicefront arecord aplay demo 2>/dev/null
sleep 1
UDC=$(cat $G/UDC)
[ -n "$UDC" ] || UDC=$(ls /sys/class/udc)
echo "" > $G/UDC
rm -f $G/configs/c.1/uac1.usb0
echo $MASK > $F/p_chmask
ln -s $F $G/configs/c.1/uac1.usb0
echo $UDC > $G/UDC
echo "gadget rebound: p_chmask=$(cat $F/p_chmask) udc=$(cat $G/UDC)"
sleep 1
[ -x /data/build/rawmic.sh ] && grep -q rawmic /etc/rc.local && start-stop-daemon -S -b -x /data/build/rawmic.sh && echo "rawmic started"
[ -x /data/build/voicefront/voicefront.sh ] && grep -q voicefront /etc/rc.local && start-stop-daemon -S -b -x /data/build/voicefront/voicefront.sh && echo "voicefront started"
[ -n "$WATCH" ] && kill -CONT $WATCH && echo "watchdog resumed (restarts demo unless voicefront disabled it)"
echo done
