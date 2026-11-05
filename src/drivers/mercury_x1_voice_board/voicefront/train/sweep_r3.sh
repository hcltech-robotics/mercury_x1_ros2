#!/bin/bash
# After round 3: retrain only (features already computed) with other model sizes / false-wake targets, in parallel,
# each in its own output dir that links the clips + features. Then eval_r3.py on each.
cd ~/oww_hey_mercury; source .venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
M=hey_mercury_model/hey_mercury
# name feats layer max_neg target_fp
VARIANTS="A:r3:128:750:0.2 B:r3:32:150:1.0 C:r2:32:150:1.0 D:r3:128:150:1.0"
for v in $VARIANTS; do
  IFS=: read n f l w t <<< "$v"
  d=sweep_$n; rm -rf $d; mkdir -p $d/hey_mercury
  for s in positive_train positive_test negative_train negative_test; do ln -s ../../$M/$s $d/hey_mercury/$s; done
  src=$M; [ $f = r2 ] && src=r2_features
  for x in positive_features_train positive_features_test negative_features_train negative_features_test; do
    ln -s $(readlink -f $src/$x.npy) $d/hey_mercury/$x.npy; done
  base=hey_mercury_r3.yaml; [ $f = r2 ] && base=hey_mercury_r2.yaml
  sed -e "s|^output_dir:.*|output_dir: ./$d|" -e "s|^layer_size:.*|layer_size: $l|" \
      -e "s|^max_negative_weight:.*|max_negative_weight: $w|" -e "s|^target_false_positives_per_hour:.*|target_false_positives_per_hour: $t|" \
      $base > $d/config.yaml
  ( python openWakeWord/openwakeword/train.py --training_config $d/config.yaml --train_model > $d/train.log 2>&1
    echo "TRAIN_EXIT=$?" >> $d/train.log
    python -W ignore eval_r3.py $d/hey_mercury.onnx > $d/eval.txt 2>&1 ) &
done
wait
for v in $VARIANTS; do n=${v%%:*}; echo "== $v"; grep -E "TRAIN_EXIT|Final Model" sweep_$n/train.log; grep -v -i warn sweep_$n/eval.txt; done
echo SWEEP_DONE
