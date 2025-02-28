FROM nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 AS builder_container
ARG HOST_TRTLLM_DIR
COPY ${HOST_TRTLLM_DIR} /workspace
WORKDIR /workspace
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt


FROM nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 AS server_container
ARG HOST_REPO_DIR
ARG CONFIG_FILL_SCRIPT
ARG RUN_SERVE_SCRIPT
WORKDIR /workspace

COPY ${HOST_REPO_DIR} ./repo
COPY ${CONFIG_FILL_SCRIPT} ./fill_template.py
COPY ${RUN_SERVE_SCRIPT} ./launch_triton_server.py

ENV ENGINE_DIR=./engine
ENV TOKENIZER_DIR=./tokenizor
ENV MODEL_FOLDER=./repo
ENV TRITON_MAX_BATCH_SIZE=2
ENV INSTANCE_COUNT=1
ENV MAX_QUEUE_DELAY_MS=0
ENV MAX_QUEUE_SIZE=0
ENV DECOUPLED_MODE=false
ENV LOGITS_DATATYPE=TYPE_FP32

RUN python3 fill_template.py -i ./repo/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}
RUN python3 fill_template.py -i ./repo/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
RUN python3 fill_template.py -i ./repo/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE}
RUN python3 fill_template.py -i ./repo/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
RUN python3 fill_template.py -i ./repo/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}

CMD python3 launch_triton_server.py --world_size=1 --model_repo=$MODEL_FOLDER
