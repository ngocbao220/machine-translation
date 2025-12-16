 parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn folder data')
    parser.add_argument('--checkpoint', type=str, required=True, help='File .pth model đã train')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='vi')
    parser.add_argument('--output_file', type=str, default='test_results.csv', help='File lưu kết quả dịch')

python helper/eval.py --data_dir="../data/processed/IWSLT15" --checkpoint="../checkpoints/base_transformer/IWSLT15/en_vi/best_model.pth"
python helper/train.py --data_dir="../data/processed/IWSLT15" --checkpoint="../checkpoints/base_transformer/IWSLT15/en_vi/best_model.pth"