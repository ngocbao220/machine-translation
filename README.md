# Dự án Machine Translation (MT)

## Giới thiệu

Dự án này là một hệ thống dịch máy (machine translation) dựa trên Transformer, chứa mã nguồn để huấn luyện, đánh giá và sinh kết quả dịch cho các tập dữ liệu khác nhau (ALT, IWSLT15, Medical).

Mục tiêu: xây dựng, huấn luyện và đánh giá các mô hình transformer cho các cặp ngôn ngữ trong thư mục `data/processed`.

## Cấu trúc chính

- `train.py` — script huấn luyện mô hình.
- `eval.py` — script đánh giá (tính BLEU/số liệu khác) trên tập test.
- `configs/` — các file cấu hình YAML cho các chạy khác nhau.
- `data/processed/` — dữ liệu đã tiền xử lý (BPE, vocab, train/test files).
- `outputs/checkpoints/` — nơi lưu checkpoint và mô hình huấn luyện.
- `scripts/` — các shell helper (ví dụ: `run_train_base.sh`).

## Yêu cầu

1. Python 3.10+ (hoặc môi trường tương thích).
2. Cài các package trong `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Các tập dữ liệu tiền xử lý đã nằm trong `data/processed/`. Nếu cần tiền xử lý lại, xem các script trong `data/` (loader) để biết định dạng đầu vào mong đợi.

Lưu ý: kiểm tra file BPE / vocab tương ứng trong mỗi bộ dữ liệu (ví dụ `data/processed/IWSLT15/bpe_model.json`).

## Huấn luyện

Ví dụ huấn luyện cơ bản sử dụng file config `configs/base_iwslt25.yaml`:

```bash
data:
  dataset_name: "ALT"
  src_lang: "en"
  tgt_lang: "vi"
  vocab_size: 10000

model:
  model_type: "base"       # Đổi thành "re" để dùng ReTransformer
  d_model: 512
  nhead: 4
  num_layers: 4
  dim_feedforward: 1024
  dropout: 0.2

train:
  batch_size: 64
  lr: 0.0005
  epochs: 30
  gpu_mode: true
  use_amp: true      # Mixed Precision
  save_dir: "outputs/checkpoints/base_alt_en_vi"
  use_wandb: true

logging:
  use_wandb: true
  project_name: "base_alt_en_vi"
  wandb_api_key: "my_key"
```

```bash
python train.py --config configs/base_iwslt25.yaml
```

Hoặc dùng script helper (nếu có):

```bash
bash scripts/run_train_base.sh
```

Một số lưu ý:

- Checkpoint sẽ được lưu vào `outputs/checkpoints/` (hoặc thư mục được chỉ định trong config).
- Các tham số huấn luyện (batch size, lr, epochs, v.v.) được điều khiển trong file YAML trong `configs/`.

## Đánh giá (Evaluation)

Sau khi có checkpoint, chạy `eval.py` để dịch và tính số liệu (ví dụ BLEU). Ví dụ:

```bash
python eval.py --config configs/base_iwslt25.yaml --checkpoint outputs/checkpoints/base_alt_en_vi/best_model.pt
```

## Kết quả thu được

| Dataset        | Size    | BLEU Score |
| :------------- | :------ | :--------- |
| **IWSLT15**    | 133,000 | **28.22**  |
| **ALT Corpus** | 20,000  | **14.68**  |
| **Medical**    | 500,000 | **35.67**  |
