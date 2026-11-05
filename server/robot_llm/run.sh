#!/bin/bash
# The robot's own language model for voice chat (failsafe / fully on board): llama.cpp's llama-server on the
# Jetson GPU, OpenAI-compatible on http://127.0.0.1:8091/v1/chat/completions (VOICE_LOCAL_LLM_URL in voice.py).
# Settings: ~/.config/mercury_x1/llm.env (LLM_MODEL=<file in ~/llm/models>, LLM_CTX, LLM_EXTRA).
D=~/llm
LLM_MODEL=Qwen3.5-4B-Q4_K_M.gguf
LLM_CTX=8192
LLM_EXTRA=""
[ -f ~/.config/mercury_x1/llm.env ] && . ~/.config/mercury_x1/llm.env
# shellcheck disable=SC2086
exec "$D/llama.cpp/build/bin/llama-server" -m "$D/models/$LLM_MODEL" --alias local \
  --host 127.0.0.1 --port 8091 -ngl 99 -c "$LLM_CTX" -np 1 -fa on --jinja --load-mode none \
  --cache-reuse 256 -t 4 $LLM_EXTRA
