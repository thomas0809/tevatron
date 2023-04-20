
#MASTER_PORT=$(shuf -n 1 -i 10000-65535)
#
#torchrun --nproc_per_node=2 --master_port ${MASTER_PORT} -m tevatron.driver.train \
#  --output_dir model_debug \
#  --train_dir preprocessed/USPTO_debug/train.jsonl \
#  --val_dir preprocessed/USPTO_debug/val.jsonl \
#  --cache_dir cache/ \
#  --model_name_or_path seyonec/ChemBERTa-zinc-base-v1 \
#  --p_model_name_or_path allenai/scibert_scivocab_uncased \
#  --do_train \
#  --dataloader_num_workers 4 \
#  --save_steps 20000 \
#  --fp16 \
#  --per_device_train_batch_size 32 \
#  --train_n_passages 1 \
#  --learning_rate 1e-5 \
#  --q_max_len 256 \
#  --p_max_len 256 \
#  --num_train_epochs 20 \
#  --negatives_x_device \
#  --overwrite_output_dir


#python -m tevatron.driver.encode \
#  --output_dir=temp \
#  --model_name_or_path model_debug \
#  --tokenizer_name model_debug/passage_model \
#  --fp16 \
#  --per_device_eval_batch_size 256 \
#  --p_max_len 256 \
#  --dataset_name csv \
#  --encode_in_path preprocessed/USPTO_debug/corpus.csv \
#  --encoded_save_path corpus_emb.pkl

#python -m tevatron.driver.encode \
#  --output_dir=temp \
#  --model_name_or_path model_debug \
#  --tokenizer_name model_debug/query_model \
#  --fp16 \
#  --per_device_eval_batch_size 256 \
#  --q_max_len 256 \
#  --dataset_name json \
#  --encode_in_path preprocessed/USPTO_debug/val.jsonl \
#  --encoded_save_path query.pkl \
#  --encode_is_qry

python -m tevatron.faiss_retriever \
  --query_reps query.pkl \
  --passage_reps corpus_emb*.pkl \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to rank.txt
