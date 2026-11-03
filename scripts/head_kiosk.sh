#!/bin/bash
# X client of the head display kiosk (xinit, started by mercury-x1-head.service on tty1): the head UI
# (web_server :8081) full screen in surf, a small WebKitGTK browser. Restarted if it ever exits.
URL="${MX1_HEAD_URL:-http://127.0.0.1:8081/}"
xset s off -dpms s noblank 2>/dev/null
xsetroot -solid "#101216" 2>/dev/null
# WebKitGTK's GPU paths (compositing, DMA-BUF renderer) are unreliable on the Tegra X driver; the page
# is a simple status UI, CPU rendering is plenty
export WEBKIT_DISABLE_COMPOSITING_MODE=1 WEBKIT_DISABLE_DMABUF_RENDERER=1 NO_AT_BRIDGE=1
# without a window manager surf stays an 800x600 window (nothing honours its full-screen request);
# matchbox is a tiny kiosk WM that shows every window full screen, without a title bar
matchbox-window-manager -use_titlebar no -use_cursor no &
until curl -sf -m 2 -o /dev/null "$URL"; do sleep 2; done      # the stack is still starting
while true; do
  # -F full screen, -K kiosk (no shortcuts / context menu), -z 1 zoom
  dbus-run-session surf -F -K -z 1 "$URL"
  sleep 2
done
