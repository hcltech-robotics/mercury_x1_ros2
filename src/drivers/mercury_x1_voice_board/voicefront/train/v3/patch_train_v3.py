"""v3 patches for openWakeWord/openwakeword/train.py (idempotent):
 1. any feature_data_files key starting with "positive" is labelled 1 (lets a config add extra positive classes,
    e.g. positive_robot / positive_r2, each with its own batch_n_per_class)
 2. the 11 h false-positive validation set is scored in chunks of 32768 windows instead of one 2.9 GB batch,
    so several trainings fit in the GPU budget at once (the callers already accumulate over batches)
"""
from pathlib import Path
p = Path.home() / "oww_hey_mercury/openWakeWord/openwakeword/train.py"
s = p.read_text()
rep = [
    ('            if key == "positive":\n                label_transforms[key] = lambda x: [1 for i in x]',
     '            if key.startswith("positive"):\n                label_transforms[key] = lambda x: [1 for i in x]'),
    ('            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),\n            batch_size=len(X_val_fp_labels)',
     '            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),\n            batch_size=32768'),
]
for a, b in rep:
    if b in s:
        print("already patched:", a[:60].strip())
        continue
    assert s.count(a) == 1, a
    s = s.replace(a, b)
    print("patched:", a[:60].strip())
p.write_text(s)
