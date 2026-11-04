#!/bin/sh
# Mercury X1: copy the raw AC107 capture (shared via dsnoop) into USB channels 0-4 of the UAC1 gadget.
# Stock ALSA tools only. The pipe ends when the host closes the stream; restart it.
trap "" HUP
while true; do
    arecord -q -D Ac107Snoop -c 8 -r 16000 -f S16_LE -t raw 2>/dev/null \
        | aplay -q -D RawToUsb -c 8 -r 16000 -f S16_LE -t raw 2>/dev/null
    sleep 1
done
