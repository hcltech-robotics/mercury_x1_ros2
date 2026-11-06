#!/bin/bash
# v3 sweep: retrain the head on round-3 features + round-2 features + the round-4 robot-path features (r4/feat),
# several variants in parallel, each in sweep_v3_<name>/ ; then eval_v3.py on each.
# VARIANTS="name:layer:max_neg:target_fp:steps:pos_robot:neg_robot:pos_r2:neg_r2[:key~path~count,...] ..."  (keys starting with positive are label 1)   (per-batch counts; 0 = class not used)
# usage: ./sweep_v3.sh "E:128:750:0.2:50000:100:60:100:50 F:128:150:1.0:50000:100:60:100:50"
cd ~/oww_hey_mercury; source .venv/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
M=hey_mercury_model/hey_mercury
VARIANTS="$1"
for v in $VARIANTS; do
  IFS=: read n l w t st pr nr p2 n2 extra <<< "$v"   # extra: key~path~count,key~path~count (optional)
  d=sweep_v3_$n; rm -rf $d; mkdir -p $d/hey_mercury
  for s in positive_train positive_test negative_train negative_test; do ln -s ../../$M/$s $d/hey_mercury/$s; done
  for x in positive_features_train negative_features_train; do ln -s $(readlink -f $M/$x.npy) $d/hey_mercury/$x.npy; done
  # checkpoint selection on the standard test clips AND the same clips through the robot path
  python - $d <<'P'
import sys, numpy as np
d = sys.argv[1]
for k in ("positive", "negative"):
    a = np.load(f"hey_mercury_model/hey_mercury/{k}_features_test.npy"); b = np.load(f"r4/feat/{k}_test_robot.npy")
    np.save(f"{d}/hey_mercury/{k}_features_test.npy", np.concatenate([a, b]))
P
  {
    sed -e "s|^output_dir:.*|output_dir: ./$d|" -e "s|^layer_size:.*|layer_size: $l|" -e "s|^steps:.*|steps: $st|" \
        -e "s|^max_negative_weight:.*|max_negative_weight: $w|" -e "s|^target_false_positives_per_hour:.*|target_false_positives_per_hour: $t|" \
        hey_mercury_r3.yaml | sed -e '/^feature_data_files:/,$d'
    echo "feature_data_files:"
    echo "  ACAV100M_sample: ./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    [ "$pr" != 0 ] && echo "  positive_robot: ./r4/feat/positive_robot.npy"
    [ "$nr" != 0 ] && echo "  negative_robot: ./r4/feat/negative_robot.npy"
    [ "$p2" != 0 ] && echo "  positive_r2: ./r2_features/positive_features_train.npy"
    [ "$n2" != 0 ] && echo "  negative_r2: ./r2_features/negative_features_train.npy"
    for e in ${extra//,/ }; do IFS='~' read k pth c <<< "$e"; echo "  $k: $pth"; done
    echo "batch_n_per_class:"
    echo "  ACAV100M_sample: 1024"
    echo "  adversarial_negative: 100"
    echo "  positive: 200"
    [ "$pr" != 0 ] && echo "  positive_robot: $pr"
    [ "$nr" != 0 ] && echo "  negative_robot: $nr"
    [ "$p2" != 0 ] && echo "  positive_r2: $p2"
    [ "$n2" != 0 ] && echo "  negative_r2: $n2"
    for e in ${extra//,/ }; do IFS='~' read k pth c <<< "$e"; echo "  $k: $c"; done
    echo "model_type: dnn"
    echo "layer_size: $l"
    echo "steps: $st"
    echo "max_negative_weight: $w"
    echo "target_false_positives_per_hour: $t"
  } > $d/config.yaml
  ( nice -n 10 python openWakeWord/openwakeword/train.py --training_config $d/config.yaml --train_model > $d/train.log 2>&1
    echo "TRAIN_EXIT=$?" >> $d/train.log
    nice -n 10 python -W ignore eval_v3.py $d/hey_mercury.onnx 2>&1 | grep -v -i warn > $d/eval.txt ) &
  sleep 20
done
wait
for v in $VARIANTS; do n=${v%%:*}; echo "== $v"; grep -E "TRAIN_EXIT|Final Model|Best model from" sweep_v3_$n/train.log | tail -4; grep SUMMARY sweep_v3_$n/eval.txt; done
echo SWEEP_V3_DONE
