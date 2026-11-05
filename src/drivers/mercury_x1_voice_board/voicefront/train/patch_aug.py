"""Round-2 augmentation for far-field/quiet mics: deeper attenuation, more reverb, lower coloured-noise SNR."""
from pathlib import Path
p = Path.home() / "oww_hey_mercury/openWakeWord/openwakeword/data.py"
s = p.read_text()
rep = [
    ('torch_audiomentations.Gain(max_gain_in_db=0, p=augmentation_probabilities["Gain"])',
     'torch_audiomentations.Gain(min_gain_in_db=-30, max_gain_in_db=0, p=augmentation_probabilities["Gain"])'),
    ('min_snr_in_db=10, max_snr_in_db=30,', 'min_snr_in_db=0, max_snr_in_db=30,'),
    ('            "RIR": 0.5\n', '            "RIR": 0.8\n'),
]
for a, b in rep:
    c = s.count(a); assert c >= 1, a
    s = s.replace(a, b); print(f"replaced x{c}: {a[:50]}")
p.write_text(s)
