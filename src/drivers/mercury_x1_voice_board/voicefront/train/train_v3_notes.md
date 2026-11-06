# hey_mercury v3 — retrained for the robot's own microphones (2026-11-05)

Deliverables (not deployed, not committed):

| File | What | Size |
|---|---|---|
| `../model/hey_mercury_v3.onnx` | **recommended**: output-average ensemble of heads I + J + K (below) | 2.5 MB |
| `../model/hey_mercury_v3_loose.onnx` | single head M, higher recall / higher false-wake rate | 859 KB |
| `v3/` | every script, config and eval printout used (also on the trainer in `~/oww_hey_mercury/export_v3`) | |

Both take the same input/output as the current model (`onnx::Flatten_0` `[1,16,96]` -> `39` `[1,1]`), so
voicefront loads them unchanged with `--oww`. The ensemble is three 128-unit DNN heads plus a `Mean` node
(3x the head cost, still negligible next to the embedding model).

## Result

Far-field = Kokoro TTS played through a monitor ~2 m from the robot, mic 0 at board level +20 dB, hum included
(`r3/ff_test`: 14 voices the model never heard, 28 positives / 12 negatives). Validation = openWakeWord's 10.7 h
negative speech/noise features. "Robot stream" = 30 min of the robot's hum/hiss with the robot talking, cross-faded
joins, x10 (`stream_xfade.py`).

| Model | far-field recall @0.5 | @0.3 | false wakes/h (10.7 h val) @0.5 | @0.3 | robot stream 30 min @0.5 / @0.3 | clean Kokoro 56 @0.5 | board-mic 5 clips @0.5 |
|---|---|---|---|---|---|---|---|
| round 2 (`export_r2`) | 10/28 | 10/28 | 0.37 | 0.37 | 0 / 0 | 45 | 3 |
| **deployed D** (sweep D, 128 units, loose target; runs at 0.3) | 14/28 | 20/28 | 1.96 | 3.18 | 0 / 0 | 54 | 4 |
| **v3 = ensemble I+J+K** | **18/28** | **18/28** | **0.28** | **0.47** | 0 / 1 | 53 | 3 |
| v3_loose = head M | 21/28 | 22/28 | 1.03 | 1.31 | 3 / 3 | 55 | 3 |
| head I (single) | 19/28 | 19/28 | 0.19 | 0.65 | 1 / 2 | 54 | 3 |
| head J (single) | 18/28 | 18/28 | 0.37 | 0.65 | 0 / 1 | 54 | 3 |
| head K (single) | 17/28 | 17/28 | 0.37 | 0.65 | 0 / 0 | 54 | 3 |

Far-field negatives: 1/12 fires for every new model and for D — it is "Hey Mercedes!" (male voice, 2 m), round 2
scored it 0.00. The remaining 11 ("hey Mark", "hey there", "Mercury is the closest planet…", …) stay below 0.05.
Seen far-field voices (the 14 voices whose recordings are in the training set): v3 15/28 mic 0, 13/28 mic 2;
D 19/28 and 15/28; r2 8/28 and 5/28. The robot's own speech (pc recordings 16–36 s) never exceeds 0.03, the pure
hum sections never exceed 0.01, and the person in those recordings scores 0.99–1.00 on every model.

Repeat runs of recipe I (I2, I3) gave 18/28 and 18/28 but 0.47 and 0.65 false wakes/h, so single heads vary by
±1 clip and ±0.3/h between seeds; the ensemble removes most of that variance, which is why it is the recommendation.

Per-voice picture (`v3/diag_scores.py` output): v3 fixes the deep male voices that round 2 scored 0.00
(am_santa, bm_fable, bm_lewis now 0.63–0.99; am_adam, am_liam second takes 0.5), leaves af_kore, af_nova, bf_lily
and am_onyx unsolved (0.01–0.09; these clips have 11–14 dB speech-band SNR and are mostly 2nd-take phrasings).

### Threshold recommendation

- `hey_mercury_v3.onnx` at **0.3** (same as today): 18/28 far-field, 0.47 false wakes/h vs the deployed 20/28 at
  3.18/h. The score distribution is bimodal (hits 0.5–1.0, misses < 0.1), so 0.3 and 0.5 give the same recall;
  0.5 gives 0.28/h. Going below 0.3 buys nothing (0.2: still 18/28, 0.84/h).
- If the Orin's speech-recognition confirmation stays in the loop and false wakes are cheap, `hey_mercury_v3_loose.onnx`
  at 0.3 gives 22/28 at 1.31/h — more recall than D with 2.4x fewer false wakes — but it is a single head and does fire
  on the hum stream 3 times in 30 min (D: 0, ensemble: 0 at 0.5).

## What was wrong with round 3 (and why GTCRN / notching could not help)

`sim_fidelity.py`: take the clean Kokoro clip, convolve with the measured monitor->mic response, add the recorded
robot noise at the real level, score with round 2 / D:

| condition | r2 hits /28 | D hits /28 |
|---|---|---|
| real far-field recording | 10 | 14 |
| clean | 21 | 26 |
| clean + robot noise only | 19 | 26 |
| clean * robot RIR only | 14 | 25 |
| clean * robot RIR + robot noise | 6 | 10 |

Neither the noise nor the room/monitor response alone hurts much; the *combination* (reverb smears the consonants,
then the hum/hiss buries what is left) is what the frozen embedding cannot cope with, and the simulation is a faithful
(slightly pessimistic) stand-in for the real recordings. Round 3 added the robot RIRs and robot noise to openWakeWord's
random augmentation independently (RIR p=0.8 of which ~half robot, background p=0.75 of which ~half robot, SNR drawn
from -10..15 dB), so only ~15 % of clips saw both at once and most of those at an unrealistic SNR. Measured real
speech/noise SNR on the far-field set: 2.6–11 dB, median 5.4 dB (`snr_real.py`).

## Recipe

Trainer: `hcltech@192.168.10.52:~/oww_hey_mercury` (`.venv`, openWakeWord fork, round-2/3 clips and features kept).
Nothing from rounds 2/3 was regenerated; v3 adds features and retrains only the head (7–15 min per head, ~700 MB GPU).

1. **Robot-path feature set** `build_r4.py` -> `r4/feat/` (13 min, 10 nice'd CPU workers + GPU embedding).
   Inputs: the 7,757 Whisper-filtered Piper positives + 32,000 random LibriTTS-R positives; 24,000 random adversarial
   negatives; the robot's own speech recorded by its mics (16–36 s of the three `mercury_mic_test` files, x12 copies);
   robot1/robot2/p2 clean clips; 2,400 pure-noise pieces. Every speech clip: 30 % speed perturbation (0.9–1.1),
   placed in a 2 s window ending 0–200 ms before the end, then **RIR** (70 % one of the 4 measured robot-mic responses,
   20 % MIT RIR, 10 % none), then **robot noise** from a 134-piece pool (`r3/robot_noise` 4 mics x 5 fan speeds, the
   silence of the three pc recordings, `fan_full`/`fan_slow` from `/tmp/mt` on the robot, 4 mics each) at
   **SNR ~ U(-2, 14) dB** measured on speech-active frames, then peak gain U(-30, -3) dB. Also the 2,000-clip
   positive/negative test sets through the same path (used for checkpoint selection) and a 30 min false-wake stream.
   `r4/feat_hard/`: same with SNR U(-6, 8) dB and robot RIR on 100 % (env `R4_SNR_LO/HI`, `R4_PROBOT`, `R4_OUT`).
2. **Pure-noise negatives** `build_noise_neg.py` -> `r4/feat/noise_neg.npy`: 8,000 two-second pieces of the robot
   noise at -42..-6 dBFS peak, 30 % spliced from two pieces, 30 % speed-shifted ±7 %, 20 % with a level ramp.
3. **train.py patch** `patch_train_v3.py`: any `feature_data_files` key starting with `positive` is label 1 (extra
   positive classes with their own batch share), and the 10.7 h validation set is scored in 32k-window chunks instead
   of one 2.9 GB GPU batch (so 4 heads train in parallel under 6 GB).
4. **Heads** `sweep_v3.sh "<name>:<layer>:<max_neg_weight>:<target_fp/h>:<steps>:<pos_robot>:<neg_robot>:<pos_r2>:<neg_r2>[:extras]"`
   Each head trains on: ACAV100M 1024 / adversarial 100 / positive (round-3 features) 200 per batch, plus the round-2
   features (`positive_r2` 100, `negative_r2` 50, i.e. a second augmentation round of every clip), plus the classes
   below; checkpoint selection on round-3 test features + the robot-path test features concatenated. 128-unit DNN.

   | head | robot-path classes per batch | noise class | steps | max neg weight | target fp/h | config |
   |---|---|---|---|---|---|---|
   | I | `positive_robot` 200, `negative_robot` 120 (feat) | 32 | 50k | 750 | 0.2 | `v3/config_I.yaml` |
   | J | `positive_robot_hard` 120+80, `negative_robot_hard` 80+40 (feat_hard only) | 32 | 50k | 750 | 0.2 | `v3/config_J.yaml` |
   | K | feat 120/80 + feat_hard 120/80 | 32 | 50k | 750 | 0.2 | `v3/config_K.yaml` |
   | M | feat 200/120 + feat_hard 60/40 | 32 | 75k | 750 | 0.3 | `v3/config_M.yaml` |

5. **Ensemble** `build_ensemble.py ens/IJK.onnx sweep_v3_I/hey_mercury.onnx sweep_v3_J/... sweep_v3_K/...`
   renames each graph's tensors and appends a `Mean` node. Output-averaging is the right merge here (openWakeWord's
   own checkpoint averaging only works within one training run).
6. **Evaluation** `eval_v3.py MODEL.onnx` (streaming 80 ms frames like voicefront; all tables above),
   `stream_xfade.py`, `diag_pieces.py`, `diag_stream.py`.

## What helped, what did not

- Helped most: the joint robot path (RIR **and** noise at the real SNR) as its own weighted class: 10 -> 17–19/28
  held-out at equal or lower false-wake rate than round 2 (G/I/J/K). Heavier weight (200 vs 100 per batch) mattered:
  E (100/60) stayed at 10–13/28.
- Helped: the pure-noise class (I vs G: 0.28 -> 0.19/h at the same recall), the round-2 features as a second
  augmentation round, output-averaging the heads (removes seed variance, 0 hits on the hum stream at 0.5).
- Neutral: the harder SNR set (J/K ≈ I); longer training / looser target (H, L, M) only trades recall for false
  wakes along the same curve (L: 23/28 at 3.65/h).
- Did not help: round 3's independent RIR/noise augmentation (7/28); GTCRN or a 250 Hz notch at inference (see
  `ab_gtcrn.py`, `ab_hum.py` from the previous session).
- Not done (time): more TTS voices / pitch variation (the far-field misses are specific voices, mostly 2nd-take
  phrasings; the quickest real gain is more real far-field positives), and moving the `kws` input to the beamformer
  (`hdmi_beam` clips score 1–2/2 on all models).

## Caveats

- The far-field set is small (28 held-out positives): differences of 1–2 clips are noise (same recipe re-run: ±1).
- The "robot stream" is synthetic: my first version had abrupt splices between noise pieces, which alone triggered the
  new heads (the `robot noise+talk stream` lines in `v3/eval_*.txt`); no single noise piece triggers any model even at
  x30 (`diag_pieces.py`), and the cross-faded stream numbers are the ones in the table.
- GPU on the trainer stayed under 5 GB with four heads training; nothing of other users was touched.
