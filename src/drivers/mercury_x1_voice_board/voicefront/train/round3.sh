#!/bin/bash
# Round 3: add the robot's acoustics (r3/ from build_assets.py), re-augment every clip, train, evaluate.
# Run on the trainer in ~/oww_hey_mercury (tmux). Round-2 features are moved to r2_features/.
set -x
cd ~/oww_hey_mercury; source .venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
M=hey_mercury_model/hey_mercury
mkdir -p r3/robot_rirs_dup r2_features
for k in 0 1 2 3; do for c in $(seq -w 1 70); do ln -sf ../robot_rirs/robot_mic$k.wav r3/robot_rirs_dup/robot_mic${k}_$c.wav; done; done
# real far-field positives x25 (each copy gets its own random augmentation)
rm -f $M/positive_train/ff_*.wav
for f in r3/ff_pos_train/*.wav; do b=$(basename $f .wav); for c in $(seq -w 1 25); do ln -f $f $M/positive_train/ff_${b}_c$c.wav; done; done
# pure robot noise as negatives: 1.5 s pieces
rm -f $M/negative_train/hum_*.wav
python - <<'P'
import glob, soundfile as sf
n = 0
for f in sorted(glob.glob("r3/robot_noise/*.wav")):
    y, sr = sf.read(f, dtype="float32")
    for i in range(0, len(y) - 24000, 24000):
        sf.write(f"hey_mercury_model/hey_mercury/negative_train/hum_{n:05d}.wav", y[i:i + 24000], sr); n += 1
print("hum negatives", n)
P
for d in positive_train positive_test negative_train negative_test; do echo "$d: $(ls $M/$d | wc -l)"; done
mv -f $M/*.npy r2_features/ 2>/dev/null
python openWakeWord/openwakeword/train.py --training_config hey_mercury_r3.yaml --augment_clips > augment_r3.log 2>&1
echo "AUG_R3_EXIT=$?"
ls -la $M/*.npy
python openWakeWord/openwakeword/train.py --training_config hey_mercury_r3.yaml --train_model > train_r3.log 2>&1
echo "TRAIN_R3_EXIT=$?"
mkdir -p export_r3; cp hey_mercury_model/hey_mercury.onnx export_r3/
python -W ignore eval_r3.py export_r2/hey_mercury.onnx > eval_r3_baseline_r2.txt 2>&1
python -W ignore eval_r3.py export_r3/hey_mercury.onnx > eval_r3.txt 2>&1
cat eval_r3_baseline_r2.txt eval_r3.txt
echo ROUND3_DONE
