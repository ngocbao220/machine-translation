print("🚀 Bắt đầu huấn luyện mô hình Dịch máy...")
print("Src: en | Tgt: vi")
print("---------------------------------------------")

import subprocess; subprocess.run([
    "python", "src/helper/train.py",
    "--data_dir", "./data/processed/ALT",
    "--base_checkpoint_dir", "./checkpoints",
    "--model_name", "transformer_base",
    "--dataset_name", "ALT",
    "--src_lang", "en",
    "--tgt_lang", "vi",
    "--batch_size", "32",
    "--lr", "0.0001",
    "--epochs", "20",
    "--vocab_size", "10000",
    "--bos_idx", "2",
    "--eos_idx", "3",
    "--add_timestamp",
    "--wandb_api_key", "4379f72f74531ca44486a34cb4e859dc0e3cf103",
    "--use_wandb"
], check=True)

print("✅ Quá trình huấn luyện kết thúc!")