#!/bin/bash
tritonserver \
  --model-repository=/workspace/repository \
  --model-control-mode=poll \
  --repository-poll-secs=5

  