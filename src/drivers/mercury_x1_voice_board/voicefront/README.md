# voicefront: our mic-array front end for the head voice board

`voicefront` replaces the board's stock iFlytek `demo`. It reads the 4 microphones and the speaker echo
reference, and produces:

- a processed beam for speech recognition, on USB audio channel 5 (`mx1_beam` on the Orin);
- an English wake word ("hey mercury" by default) with its direction;
- a continuous direction of arrival while someone speaks;
- voice activity events.

Events and commands travel as JSON lines over the board's UART, which reaches the Orin as `/dev/xf_mic`.
The same source runs on the Orin for development. It is one static aarch64 binary that runs unchanged on the
board's Tina Linux, which has glibc 2.23.

## Pipeline (per 10 ms, 16 kHz)

1. **Echo cancellation.** WebRTC AEC3 runs on all 4 mics against the reference. Its tuning (`--aec-tuning asr`)
   is a little more transparent to a person talking over the robot than WebRTC's call tuning.
2. **STFT.** The framing is 512 samples with a 256-sample hop and a sqrt-Hann window.
3. **Voice activity.** It compares the 300 to 3500 Hz band energy against a tracked noise floor.
4. **Direction.** SRP-PHAT (steered response power with phase transform) runs over 0 to 180° in 2° steps.
   It is smoothed over speech frames, and it is not updated while the speaker plays.
5. **Beamformer.** Delay-and-sum, or MVDR (`--beamformer mvdr`, minimum variance distortionless response).
   It follows the talker, and after a wake word it holds that direction for `--wake-lock-s` seconds.
6. **Output.** WebRTC noise suppression and AGC2 on the beam, then stdout or `--out-cmd`.
7. **Wake word.** sherpa-onnx keyword spotting (a 3.3M-parameter zipformer trained on GigaSpeech, int8). Its
   input is the echo-cancelled mic 0, without noise suppression or AGC. That hears every direction equally,
   and noise suppression measurably hurt recall. The wake direction is taken from the keyword's own frames.

**Geometry.** The mics form a line with 35 mm spacing. 0° points to the mic-3 end, 90° is broadside and 180°
points to the mic-0 end. A linear array can't tell front from back. The robot facing a talker reads about 90°.

## Build (on the Orin, aarch64)

```bash
./build_deps.sh      # once: WebRTC audio processing 2.1 + abseil (static), sherpa-onnx 2.0.1 static libs,
                     # keyword model -> model/, then build/voicefront (fully static)
cmake --build build  # afterwards
```

`apm.cpp` creates AEC3 itself through the internal header from the WebRTC source tree. That is needed because
the stock APM in 2.1 never sets `filter.export_linear_aec_output`, so asking for the linear output crashes.
It also gives us the tuning hook. `model/bpe.vocab` is the model's sentencepiece vocabulary, exported from its
`bpe.model`. voicefront tokenizes plain-text keywords with it, using unigram Viterbi like sentencepiece.

## Run

On the Orin, from the 6-channel USB stream:

```bash
arecord -D mx1_array -c 6 -r 16000 -f S16_LE -t raw | ./build/voicefront --channels 6 --ref 4 > beam.raw
```

Events go to stderr when there is no `--serial`.

On the board, the runner `../board/voicefront.sh` does this:

```bash
arecord -D Ac107Snoop -c 8 ... | voicefront --channels 8 --mics 0,1,2,3 --ref 6 --serial /dev/ttyS2 \
    --out-cmd "aplay -D PlaybackDmix ..."
```

`--out-cmd` restarts `aplay` when it exits. aplay exits whenever the Orin stops reading the USB audio, and
the wake word must keep working. A file on stdin is processed as fast as possible, with the keyword spotter
inline. `voicefront --help` lists all options.

## Install on the board / revert (run on the robot)

```bash
scripts/install_voicefront.sh           # build if needed, copy, disable the demo watchdog, start (+ at boot)
scripts/install_voicefront.sh revert    # stock demo back (its watchdog restarts it)
```

Board files:

| Path | What |
|---|---|
| `/data/build/voicefront/` | binary, `model/`, `voicefront.sh`, `voicefront.conf` |
| `/data/build/wakeupCron.sh` | the no-op watchdog; `.stock` is the original |
| `/etc/rc.local` | starts `voicefront.sh` |
| `/tmp/voicefront.log` | log |

`voicefront.conf` sets the AC107 input gains and extra options. `killall voicefront` applies it, because
the runner rereads it on every restart.

**Input gains.** At the stock 30 dB, the robot's own voice at full speaker volume clipped all 4 mics, and echo
cancellation can't undo clipping. The defaults are `MIC_GAIN=4` and `REF_GAIN=20`. The AC107 controls act
weaker than their dB labels, about 1.5 dB per step near the bottom of the range.

## Protocol (UART 115200 8N1, one JSON object per line)

Board to host:

| `type` | Fields |
|---|---|
| `hello` | `version`, `mics`, `rate`, `aec`, `beamformer`, `keywords`. Sent only when voicefront starts |
| `wake` | `keyword`, `angle` (deg, or null), `conf`, `playback` (the speaker was playing), `t` (s of audio) |
| `doa` | `angle`, `conf`, `beam`, `t`. During speech, `doa_hz` per second |
| `vad` | `speech` (bool), `t` |
| `status` | `cpu` (% of one core), `mode`, `beam`, `beamformer`, `speech`, `playback`, `snr_db`, `wakes`, `dropped`, `erle_db`, `keywords`, `kw_threshold`. Every `status_s` |
| `reply` | `cmd`, `ok`, `id` (echoed), `error`, plus command-specific fields |

Host to board. Every command may carry an `id`, which the reply echoes.

| `cmd` | Fields |
|---|---|
| `ping` | none. The reply carries `version` and `keywords` |
| `status` | none |
| `set_beam` | `mode`: `auto` or `fixed`, `angle` |
| `set_keywords` | `keywords`: plain English list, `threshold`, `score` |
| `set_gain` | `mic`, `ref`: AC107 PGA 0 to 31 |
| `set` | any of `beamformer`, `vad_snr_db`, `doa_hz`, `status_s`, `wake_lock_s`, `doa_during_playback` |

The ROS node `voice_board_node` detects this protocol automatically, and also still speaks the stock AIUI
protocol. It maps:

- `wake` to `/voice_board/wake`;
- every line to `/voice_board/events`;
- `set_beam`, where the beam number is an angle in degrees and -1 means auto;
- `set_wake_word` (`;`-separated English);
- `mic_vol` and `ref_vol` to `set_gain`.

## Measured (2026-11-04)

| Test | Result |
|---|---|
| Synthetic direction, talkers at 30 to 150° | median error 0.1 to 0.3°, 90th percentile ≤ 1° (≤ 2.7° at −38 dBFS noise) |
| Synthetic wake word, 36 positives (6 Kokoro voices × 6 phrasings/speeds), 65 s of other speech | 78 to 81 % hits, 0 false wakes (negatives include "Mercury is the smallest planet", "Hey Mark", "Hey there") |
| Robot says "hey mercury" through its own speaker, echo cancellation on (Orin and board) | 0 self-wakes out of 7 and 5; echo removed 19 to 35 dB |
| Same recording without echo cancellation, speaker as talker | 4 of 7 detected at about 91°; on the board 4 of 4 through UART to ROS |
| CPU, Orin (1.42 GHz cap) | 17 % of one core |
| CPU, board (Cortex-A53) | about 70 to 100 % of one core of 4, wake word on its own thread; the demo used about 25 % of one core |
| Board reboot | comes back by itself: voicefront, gains, 6 USB channels, ROS node |

## Measured (2026-11-05): hum, mic gain, echo

Test: a person's sentences from a monitor speaker 2 m away, then the robot's own voice (Kokoro, speaker at 100 %),
then both; levels in dBFS RMS on the Orin-side clean stream (`mx1_clean`, GTCRN), transcripts by Parakeet.

| Change | Silence | Person | Robot's own voice left over | Notes |
|---|---|---|---|---|
| start of day: board beam, WebRTC NS, AGC | −42..−25 | −35 | −23 | words lost |
| GTCRN on the Orin, AEC "call", AGC (PGA 4) | −56 | −25 | −48 | transcript exact |
| + `--hum` (mics only) | | | 4 dB lower | AEC converges better without the 250 Hz line |
| + mic PGA 4 → 0 (robot voice peaks −9.8 instead of −1.8 dBFS: no clipping) | | | 15 dB lower | high-band SNR 10 dB lower (ADC floor), transcript still exact |
| + `--no-agc --out-gain-db 20` (final) | **−60** | **−39** | **−69** | the AGC lifted the echo residual as much as speech |

- The hum: a cluster 246..255 Hz, 55..65 % of all noise power, 93..97 % of it below 300 Hz. `--hum` takes 9 dB
  out of the hum band and 4 dB out of the floor for 1.5 dB of speech in 200..300 Hz. Filtering the reference
  too (any filter, even a 100 Hz high-pass alone) broke AEC3: echo suppression fell from 30 dB to 6.
- Speaker volume 100 → 80 % did not stop the mic clipping at PGA 4 (peak −1.9 dBFS): the mics sit next to the
  speaker. PGA is the lever: ~2 dB per step.
- **Speaking over the robot stays unrecoverable**: at the mic the person is 28 dB under the robot's voice; with
  every AEC tuning (call / asr / light, AGC on or off) the double-talk transcript is empty or garbage.
  Interrupting the robot would have to be detection (near-end energy in the AEC output), not transcription.
- GTCRN on the board (A53, 1008 MHz, no cpufreq): through ONNX Runtime 13.9..17.7 ms per 16 ms hop = 134..157 %
  of real time for the whole pipeline; the model is ~1 MFLOP/frame but 415 nodes, ~4 ms is ORT dispatch alone,
  threads make it worse, int8 does not help (overhead, not arithmetic). A hand-written forward pass is the only
  route.

## Known limits / next

- **Wake recall on real voices is not measured yet.** The real recordings were only TTS through a monitor
  speaker and through the robot's speaker. The 3.3M keyword model is the weak part. If people find it too
  deaf, the next step is a custom-trained "hey mercury" model, for example openWakeWord, which runs on the
  ONNX Runtime already linked in.
- **Speaking over the robot is weak.** A wake word said while the robot talks is mostly missed in synthetic
  tests, because AEC3 suppresses the talker during double talk.
- **Direction sign is not calibrated.** The direction sign (`angle_sign` in `voice_board.yaml`) needs the
  base-rotation test with an external speaker.
- **Restart glitch.** While voicefront starts (model load), the raw-mic relay `rawmic.sh` can underrun for a
  few seconds, and USB channels 0 to 4 are zero during that time.

Test tools are in `test/`: `make_synthetic.py` makes array signals, and `eval_kws.py` measures wake-word
recall and false wakes.
