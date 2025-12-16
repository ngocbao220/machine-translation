import re
import os

# Config
DATA_DIR = "../../data/raw/ALT-Parallel-Corpus-20191206"
TARGET_DIR = "../../data/processed/ALT-Parallel-Corpus-20191206"

def get_txt_file_name(data_dir):
    return [f for f in os.listdir(data_dir) 
            if f.endswith(".txt") and f.startswith("data")]

def clean_data(data_dir, tgt_dir):
    file_names = get_txt_file_name(data_dir=data_dir)
    os.makedirs(tgt_dir, exist_ok=True)

    for fn in file_names:
        sentence = []
        # Read src file
        src_path = f"{data_dir}/{fn}"

        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line = line.strip().split("\t")
                    if len(line) == 2:
                        main_text = line[1]
                        sentence.append(main_text)
        
        # Write to target file
        tgt_path = f"{tgt_dir}/{fn}"
        with open(tgt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentence))


if __name__ == "__main__":
    file_names = get_txt_file_name(data_dir=DATA_DIR)

    print(f"All file names in {DATA_DIR}: {file_names}")

    clean_data(data_dir=DATA_DIR, tgt_dir=TARGET_DIR)