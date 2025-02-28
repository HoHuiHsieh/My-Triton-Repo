IGNORE_CKPT=false
MODEL_DIR="./model"
CKPT_DIR="./ckpt"
OUTPUT_DIR="./engine"

# parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ignore_ckpt) IGNORE_CKPT=true ;;
        --model_dir) MODEL_DIR="$2"; shift ;;
        --ckpt_dir) CKPT_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        -h|--help) 
            echo "Usage: $0 [--ignore_ckpt] [--model_dir <path>] [--ckpt_dir <path>] [--output_dir <path>]"
            exit 0
            ;;
        *) 
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [--ignore_ckpt] [--model_dir <path>] [--ckpt_dir <path>] [--output_dir <path>]"
            exit 1
            ;;
    esac
    shift
done

# build checkpoint
if [ "$IGNORE_CKPT" = false ]; then
    python3 convert_checkpoint.py --model_dir $MODEL_DIR \
                                  --output_dir $CKPT_DIR \
                                  --dtype float16 
fi

# build trtllm model
trtllm-build --checkpoint_dir $CKPT_DIR \
                --output_dir $OUTPUT_DIR \
                --gemm_plugin auto \
                --context_fmha disable \
                --max_batch_size 2