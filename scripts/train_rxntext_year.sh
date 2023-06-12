
MODEL_DIR=output/condition_year_b512_ep50
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

python -m torch.distributed.launch --nproc_per_node=8 --master_port $MASTER_PORT -m tevatron.driver.train \
  --output_dir ${MODEL_DIR} \
  --train_dir preprocessed/USPTO_condition_year/train_rn.jsonl \
  --cache_dir cache/ \
  --data_cache_dir cache/data/ \
  --model_name_or_path seyonec/ChemBERTa-zinc-base-v1 \
  --p_model_name_or_path allenai/scibert_scivocab_uncased \
  --do_train \
  --dataloader_num_workers 4 \
  --dataset_proc_num 64 \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 2 \
  --learning_rate 1e-4 \
  --q_max_len 256 \
  --p_max_len 256 \
  --num_train_epochs 50 \
  --negatives_x_device


python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT -m tevatron.driver.encode \
  --output_dir=${MODEL_DIR} \
  --cache_dir cache/ \
  --data_cache_dir cache/data/ \
  --model_name_or_path ${MODEL_DIR} \
  --tokenizer_name ${MODEL_DIR}/passage_model \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --p_max_len 256 \
  --dataset_name json \
  --encode_in_path preprocessed/USPTO_condition_year/corpus.jsonl \
  --encoded_save_path ${MODEL_DIR}/corpus.pkl


for split in val test train
do
  echo $split
  python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT -m tevatron.driver.encode \
    --output_dir=${MODEL_DIR} \
    --cache_dir cache/ \
    --data_cache_dir cache/data/ \
    --model_name_or_path ${MODEL_DIR} \
    --tokenizer_name ${MODEL_DIR}/query_model \
    --fp16 \
    --per_device_eval_batch_size 256 \
    --q_max_len 256 \
    --dataset_name json \
    --encode_in_path preprocessed/USPTO_condition_year/${split}.jsonl \
    --encoded_save_path ${MODEL_DIR}/${split}.pkl \
    --encode_is_qry

  python -m tevatron.faiss_retriever \
    --query_reps ${MODEL_DIR}/${split}.pkl \
    --passage_reps ${MODEL_DIR}/corpus.pkl \
    --depth 20 \
    --batch_size -1 \
    --save_json \
    --save_ranking_to ${MODEL_DIR}/${split}_rank.json

done


python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT -m tevatron.driver.encode \
  --output_dir=${MODEL_DIR} \
  --cache_dir cache/ \
  --data_cache_dir cache/data/ \
  --model_name_or_path ${MODEL_DIR} \
  --tokenizer_name ${MODEL_DIR}/passage_model \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --p_max_len 256 \
  --dataset_name json \
  --encode_in_path preprocessed/USPTO_condition/corpus.jsonl \
  --encoded_save_path ${MODEL_DIR}/corpus_full.pkl

for split in val test
do
  echo $split

  python -m tevatron.faiss_retriever \
    --query_reps ${MODEL_DIR}/${split}.pkl \
    --passage_reps ${MODEL_DIR}/corpus_full.pkl \
    --depth 20 \
    --batch_size -1 \
    --save_json \
    --save_ranking_to ${MODEL_DIR}/${split}_rank_full.json

done
