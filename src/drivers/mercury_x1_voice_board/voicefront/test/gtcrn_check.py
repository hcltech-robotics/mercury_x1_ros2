"""Compare the native GTCRN (gtcrn_test binary) with onnxruntime on a wav, voicefront framing (512 sqrt-Hann, hop
256, x10 input gain). Writes the ORT input/output frames, runs the binary, reports SNR per spectrum and on the
resynthesised time signal.

usage: gtcrn_check.py gtcrn.onnx gtcrn.bin gtcrn_test in.wav [workdir]
needs: numpy, onnxruntime (uv venv + uv pip install onnx onnxruntime numpy)
"""
import os, subprocess, sys, wave
import numpy as np
import onnxruntime as ort

onnx_path, bin_path, exe, wav = sys.argv[1:5]
work = sys.argv[5] if len(sys.argv) > 5 else "/tmp"
GAIN = 10.0
with wave.open(wav) as w:
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if w.getnchannels() > 1:
        x = x.reshape(-1, w.getnchannels())[:, 0].copy()
N, H = 512, 256
win = np.sqrt(np.hanning(513)[:512]).astype(np.float32)
xp = np.concatenate([np.zeros(N - H, np.float32), x, np.zeros(N, np.float32)])
nfr = (len(xp) - N) // H
frames = np.stack([np.fft.rfft(xp[f * H:f * H + N] * win) for f in range(nfr)]).astype(np.complex64) * GAIN
mix = np.stack([frames.real, frames.imag], -1).astype(np.float32)  # [nfr, 257, 2]

so = ort.SessionOptions(); so.intra_op_num_threads = 1
s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
caches = [np.zeros((2, 1, 16, 16, 33), np.float32), np.zeros((2, 3, 1, 1, 16), np.float32), np.zeros((2, 1, 33, 16), np.float32)]
ref = np.zeros_like(mix)
for f in range(nfr):
    enh, *caches = s.run(None, {"mix": mix[f:f + 1, :, None, :], "conv_cache": caches[0], "tra_cache": caches[1], "inter_cache": caches[2]})
    ref[f] = enh[0, :, 0, :]
ref_state = np.concatenate([c.ravel() for c in caches])

fin, fout = os.path.join(work, "gtcrn_in.f32"), os.path.join(work, "gtcrn_out.f32")
mix.tofile(fin)
print(subprocess.run([exe, bin_path, fin, fout, "5"], check=True, capture_output=True, text=True).stdout.strip())
raw = np.fromfile(fout, np.float32)
out = raw[:mix.size].reshape(mix.shape)
state = raw[mix.size:]

def snr(a, b):
    return 10 * np.log10((a ** 2).sum() / (((a - b) ** 2).sum() + 1e-30))

print(f"spectra: SNR vs ORT {snr(ref, out):.1f} dB, max abs err {np.abs(ref - out).max():.3e} (ref max {np.abs(ref).max():.2f})")
print(f"state:   max abs err {np.abs(ref_state - state).max():.3e} (ref max {np.abs(ref_state).max():.2f})")

def synth(spec):
    y = np.zeros(len(xp), np.float32)
    for f in range(nfr):
        Y = (spec[f, :, 0] + 1j * spec[f, :, 1]) / GAIN
        y[f * H:f * H + N] += np.fft.irfft(Y, N).astype(np.float32) * win
    return y[N - H:N - H + len(x)]

yr, yo = synth(ref), synth(out)
seg = slice(8000, None)
print(f"time signal: SNR vs ORT {snr(yr[seg], yo[seg]):.1f} dB; ORT output vs input (noise removed) {snr(x[seg], yr[seg]):.1f} dB")
y16 = np.clip(yo * 32768, -32768, 32767).astype(np.int16)
with wave.open(os.path.join(work, "gtcrn_native.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(y16.tobytes())
