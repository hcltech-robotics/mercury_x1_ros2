#!/usr/bin/env python3
"""Make the fp16 Parakeet TDT 0.6B v2 model for the robot's GPU (run on a PC: needs ~6 GB RAM, the robot has not).

  uv run --no-project --with onnx --with onnxconverter-common --with huggingface_hub \
      server/robot_speech/make_parakeet_fp16.py /tmp/parakeet
  rsync -a /tmp/parakeet/fp16/ elephant@<robot>:mercury-speech/models/parakeet-tdt-0.6b-v2/

Why not the published int8 model: ORT's int8 kernels are CPU kernels; on CUDA it ran out of memory.
Conversion notes (fp32 istupakov/parakeet-tdt-0.6b-v2-onnx -> fp16, I/O stays fp32):
  * the encoder is > 2 GB: shape inference must be file based (infer_shapes_path), else the converter
    leaves mixed-type nodes ("Add bound to different types");
  * three Casts in the pre-encoder come out mistyped: kept fp32;
  * LayerNormalization (120 of them) overflows in fp16 on CUDA -> empty transcripts: kept fp32
    (the CPU kernels compute in fp32, so the broken model looks fine on a CPU).
"""
from __future__ import annotations

import os
import sys

import onnx
from huggingface_hub import snapshot_download
from onnx import shape_inference
from onnxconverter_common import float16

REPO = "istupakov/parakeet-tdt-0.6b-v2-onnx"
KEEP_NODES = ["/pre_encode/Cast", "/pre_encode/Cast_1", "/pre_encode/Cast_2"]
KEEP_OPS = list(float16.DEFAULT_OP_BLOCK_LIST) + ["LayerNormalization"]


def main(d: str) -> None:
    src, out = os.path.join(d, "fp32"), os.path.join(d, "fp16")
    snapshot_download(REPO, local_dir=src, allow_patterns=["encoder-model.onnx", "encoder-model.onnx.data",
                                                           "decoder_joint-model.onnx", "config.json", "vocab.txt"])
    os.makedirs(out, exist_ok=True)
    inferred = os.path.join(src, "encoder-model.inferred.onnx")
    if not os.path.exists(inferred):
        shape_inference.infer_shapes_path(os.path.join(src, "encoder-model.onnx"), inferred)
    enc = float16.convert_float_to_float16(onnx.load(inferred), keep_io_types=True, disable_shape_infer=True,
                                           node_block_list=KEEP_NODES, op_block_list=KEEP_OPS)
    onnx.save(enc, os.path.join(out, "encoder-model.fp16.onnx"), save_as_external_data=True,
              all_tensors_to_one_file=True, location="encoder-model.fp16.onnx.data")
    del enc
    dec = float16.convert_float_to_float16(onnx.load(os.path.join(src, "decoder_joint-model.onnx")), keep_io_types=True)
    onnx.save(dec, os.path.join(out, "decoder_joint-model.fp16.onnx"))
    for f in ("config.json", "vocab.txt"):
        with open(os.path.join(src, f), "rb") as a, open(os.path.join(out, f), "wb") as b:
            b.write(a.read())
    for f in sorted(os.listdir(out)):
        print(f"{f:36s} {os.path.getsize(os.path.join(out, f)) >> 20:6d} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "parakeet")
