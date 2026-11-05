# Training the "hey mercury" wake-word model

`../model/hey_mercury.onnx` is an openWakeWord classifier (~200 KB, input `[1, 16, 96]` speech embeddings,
output a 0..1 score). voicefront loads it with `--oww`. The two feature models it sits on
(`melspectrogram.onnx`, `embedding_model.onnx`) are openWakeWord's own and come from `build_deps.sh`.

Trained 2026-11-04 on the RTX PRO 6000 box (hcltech@192.168.10.52, `~/oww_hey_mercury`), about 4 h in total.
Results: `metrics.txt`.

## Recipe (openWakeWord `train.py`, three steps)

1. **Positives** (`--generate_clips`): Piper's LibriTTS-R multi-speaker generator says "hey mercury",
   "hey, mercury", "hey mercury!", "hey murcury" 100k times. `gen_piper_voices.py` adds 10 more single-voice
   Piper models (US / British); `filter_piper_pos.py` keeps only clips Whisper hears as "hey m…rc…ry"
   (7757 of 12100).
2. **Adversarial negatives**: the same generator says the `custom_negative_phrases` in the YAML
   ("hey mark", "mercury", "the planet mercury", "hey siri", ...).
3. **Augmentation** (`--augment_clips`): MIT room impulse responses, AudioSet + FMA background,
   made harder for far field by `patch_aug.py` (gain down to -30 dB, reverb on 80 %, coloured noise to 0 dB SNR).
   Then every clip becomes speech-embedding features.
4. **Training** (`--train_model`): a 32-unit DNN, 50k steps, batches of 1024 ACAV100M (2000 h of general
   audio, precomputed features), 100 adversarial, 200 positive; negative weight ramps up (max 750) towards
   0.2 false wakes / hour on the 11 h validation set.

## Files

| File | Purpose |
|---|---|
| `setup_env.sh` | Python 3.11 venv (uv), torch cu128, openWakeWord + piper-sample-generator editable |
| `download.sh`, `download2.sh`, `pdl.sh` | TTS checkpoint, ACAV100M + validation features (parallel chunks), RIRs, noise |
| `convert_audioset.py`, `convert_noise.py`, `extract_partial_fma.py` | background noise to 16 kHz wav |
| `patch_piper.py`, `generate_samples.py` | piper-sample-generator v3 compatibility (16 kHz output, old import path) |
| `patch_aug.py` | stronger far-field augmentation |
| `gen_piper_voices.py`, `filter_piper_pos.py`, `tts_check.py` | extra voices, Whisper filter, pronunciation check |
| `hey_mercury.yaml` (round 1), `hey_mercury_r2.yaml` (final), `hey_mercury_r2_gen.yaml` | trainer configs |
| `round2c.sh` | round 2 chain: filter, augment, train |
| `eval_clips.py` | scores on real robot recordings + false wakes / hour |

Version pins that mattered: torch / torchaudio 2.8.0 (later releases dropped `torchaudio.load` / `info`),
SciPy < 1.15 (for `acoustics`), `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the checkpoints.

## Round 3 and the sweep (2026-11-05)

`hey_mercury_r3.yaml` / `round3.sh` add the robot's own acoustics (`build_r3_assets.py`: measured monitor-to-mic
responses, fan noise, real far-field positives); `sweep_r3.sh` retrains with other sizes / false-wake targets.
Recall only rose together with false wakes (`metrics.txt`), so the deployed model is sweep D (128 units, a loose
false-wake target) at a low threshold, and the Orin confirms each wake with speech recognition instead
(`mercury_x1_web/wake_verify.py`). `eval_r3.py` scores a model on the far-field recordings.

## Next, if it is too deaf

Add real far-field positives: people saying "hey mercury" 1 to 5 m from the robot, recorded on the board
(`arecord -D Ac107Snoop -c 8`, channel 0), or the synthetic positives replayed through a speaker and
re-recorded by the robot's mics. Then retrain from step 3.
