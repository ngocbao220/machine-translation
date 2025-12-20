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
python eval.py --config configs/base_iwslt25.yaml --checkpoint outputs/checkpoints/checkpoint_last.pt
```


## Kết quả thu được

- Dataset: IWSLT15
- BLEU (sacrebleu):  XX.X

- Dataset: ALT
- BLEU (sacrebleu):  XX.X

- Dataset: Medical
- BLEU (sacrebleu):  XX.X

