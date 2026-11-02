# mercury_x1_voice_board

Driver for the Mercury X1 head voice board: an iFlytek AIUI far-field mic-array board (Allwinner R818,
firmware "XFM-DP-V0.0.18", USB `2207:0001`). The board runs wake word, direction of arrival, beamforming and
noise suppression itself and gives the Orin:

- **processed 16 kHz mono audio** as a plain USB microphone — ALSA `plughw:CARD=XFMDPV0018` (used by the
  web voice chat; not touched by this node);
- a **control UART** (WCH CH343 `1a86:55d4`, serial `5A37020735`, `/dev/ttyACM3`) with iFlytek's AIUI frame
  protocol — this node.

The speaker is a separate C-Media USB card on the Orin; the board does not drive it.

## Run

```bash
scripts/install_voice_board_udev.sh          # once, on the robot: /dev/xf_mic (mode 0666)
ros2 launch mercury_x1_voice_board voice_board.launch.py   # also started by robot.launch.py voice_board:=true
```

## Interface

| Name | Type | |
|---|---|---|
| `/voice_board/wake` | `mercury_x1_msgs/VoiceWake` | keyword, `angle_deg` (linear array 0..180), `yaw` [rad, + left], beam, score |
| `/voice_board/state` | `mercury_x1_msgs/VoiceBoardState` | 1 Hz: port open, alive, firmware version, counters |
| `/voice_board/events` | `std_msgs/String` | every board event as JSON |
| `/voice_board/command` | `mercury_x1_msgs/srv/VoiceBoardCommand` | any control JSON, returns the board's reply |
| `/voice_board/set_wake_word` | `mercury_x1_msgs/srv/SetWakeWord` | pinyin with tones, `;`-separated for several |
| `/voice_board/set_beam` | `mercury_x1_msgs/srv/SetBeam` | lock the output beam (mic4: 0..2) |

Parameters: [`config/voice_board.yaml`](config/voice_board.yaml) (`mic_vol`, `ref_vol`, `beam` are applied
after every connect and board restart; `angle_front_deg` / `angle_sign` map the board angle to `yaw`).

```bash
ros2 service call /voice_board/command mercury_x1_msgs/srv/VoiceBoardCommand '{json: "{\"type\":\"version\"}"}'
ros2 service call /voice_board/set_beam mercury_x1_msgs/srv/SetBeam '{beam: 1}'
ros2 service call /voice_board/set_wake_word mercury_x1_msgs/srv/SetWakeWord \
  '{keyword: "xiao3 wei1 xiao3 wei1", threshold: "900", reload: true}'
ros2 param set /voice_board mic_vol 24
```

## Protocol

`A5 01 type len(LE16) id(LE16) payload checksum`, checksum = `(~sum + 1) & 0xFF`. Board → host: `0x01`
handshake (resent until confirmed), `0x04` event JSON; host → board: `0x05` control JSON; both: `0xFF`
confirm with the same id. Every control message is answered by a confirm plus an event
`{"code":0,"content":"…","type":"<request type>"}`; the board app announces itself with a `started` event.
Commands: `version`, `manual_wakeup {beam}`, `mic_vol` / `ref_vol {volume 0..30}`, `output_vol {0..7}`,
`wakeup_keywords {keyword, threshold}` (the board compiles the model itself), `setting [{group,key,value}]`
(edits the board's `user_cfg.ini`, which makes its app re-initialise), `switch_mic`, `dump_audio`, `clean_pcm`.

Notes: each message is logged to the board's `/data/build/demo.log` on its eMMC (hence the slow heartbeat).
`set_wake_word` + `reload` and the exact wake-event fields are not yet verified on the real board.
Full analysis of the board: `~/Documents/reverse_engineer/mercury_x1/voice_board/HOW_IT_WORKS.md`.

Tests (no ROS needed): `PYTHONPATH=. python3 -m pytest test/`
