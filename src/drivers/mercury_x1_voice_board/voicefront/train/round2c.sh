#!/bin/bash
# Round 2 (cont.): wait for Piper generators, whisper-filter, augment, train
set -x
cd ~/oww_hey_mercury; source .venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
while [ $(grep -L "positives added\|Traceback" extra_pos_*.log | wc -l) -gt 0 ]; do sleep 10; done
grep -h "positives added" extra_pos_*.log; grep -l Traceback extra_pos_*.log
python -W ignore filter_piper_pos.py 2>&1 | grep -v Warn > filter_piper.log; cat filter_piper.log | tail -12
for d in positive_train positive_test negative_train negative_test; do echo "$d: $(ls hey_mercury_model/hey_mercury/$d | wc -l)"; done
rm -f hey_mercury_model/hey_mercury/*.npy
python openWakeWord/openwakeword/train.py --training_config hey_mercury_r2.yaml --augment_clips > augment_r2.log 2>&1
echo "AUG_R2_EXIT=$?"
ls -la hey_mercury_model/hey_mercury/*.npy
python openWakeWord/openwakeword/train.py --training_config hey_mercury_r2.yaml --train_model > train_r2.log 2>&1
echo "TRAIN_R2_EXIT=$?"
