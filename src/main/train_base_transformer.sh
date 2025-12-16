
DATA_DIR="/kaggle/input/iwslt25"
CHECKPOINT_DIR="./checkpoint"
MODEL_NAME="transformer_base"
DATASET_NAME="IWSLT15"
SRC_LANG="en"
TGT_LANG="vi"

BATCH_SIZE=32       # Giảm xuống nếu bị OOM (Out of Memory)
LR=0.0001           # Learning Rate (thường để 1e-4 cho Transformer)
EPOCHS=20           # Số vòng lặp
VOCAB_SIZE=10000    # Kích thước từ điển BPE

BOS_IDX=2
EOS_IDX=3

USE_WANDB="--use_wandb" 
TIMESTAMP="--add_timestamp"
NO_GPU="--no_gpu" 

echo "🚀 Bắt đầu huấn luyện mô hình Dịch máy..."
echo "Src: $SRC_LANG | Tgt: $TGT_LANG"
echo "Batch Size: $BATCH_SIZE | Epochs: $EPOCHS"
echo "---------------------------------------------"

python helper/train.py \
    --data_dir "$DATA_DIR" \
    --base_checkpoint_dir "$CHECKPOINT_DIR" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --src_lang "$SRC_LANG" \
    --tgt_lang "$TGT_LANG" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --vocab_size $VOCAB_SIZE \
    --bos_idx $BOS_IDX \
    --eos_idx $EOS_IDX \
    $USE_WANDB \
    $TIMESTAMP \
    $NO_GPU

echo "✅ Quá trình huấn luyện kết thúc!"