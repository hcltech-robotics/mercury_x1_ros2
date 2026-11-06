import sys, numpy as np, onnx, onnxruntime as ort
def sess(p):
    mo = onnx.load(p)
    for v in list(mo.graph.input) + list(mo.graph.output): v.type.tensor_type.shape.dim[0].dim_param = "N"
    return ort.InferenceSession(mo.SerializeToString(), providers=["CPUExecutionProvider"])
for name, p in (("r2", "export_r2/hey_mercury.onnx"), ("r3", "export_r3/hey_mercury.onnx"), ("D", "sweep_D/hey_mercury.onnx")):
    s = sess(p); inp = s.get_inputs()[0].name
    for k in ("positive_features_test", "negative_features_test"):
        a = np.load(f"hey_mercury_model/hey_mercury/{k}.npy"); b = np.load(f"r4/feat/{k.replace('_features_test', '_test_robot')}.npy")
        pa = s.run(None, {inp: a})[0].ravel(); pb = s.run(None, {inp: b})[0].ravel()
        print(f"{name} {k:24s} standard-aug: >=0.5 {np.mean(pa>=.5):.3f} >=0.3 {np.mean(pa>=.3):.3f} | robot-path: >=0.5 {np.mean(pb>=.5):.3f} >=0.3 {np.mean(pb>=.3):.3f}")
