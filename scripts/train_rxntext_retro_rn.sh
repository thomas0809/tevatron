
MODEL_DIR=output/rxntext_retro_rn_b512_ep400
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

python -m torch.distributed.launch --nproc_per_node=8 --master_port $MASTER_PORT -m tevatron.driver.train \
  --output_dir ${MODEL_DIR} \
  --train_dir preprocessed/USPTO_50K/train_matched_rn.jsonl \
  --cache_dir cache/ \
  --data_cache_dir cache/data/ \
  --model_name_or_path seyonec/ChemBERTa-zinc-base-v1 \
  --p_model_name_or_path allenai/scibert_scivocab_uncased \
  --do_train \
  --dataloader_num_workers 4 \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 2 \
  --learning_rate 1e-4 \
  --q_max_len 128 \
  --p_max_len 256 \
  --num_train_epochs 400 \
  --negatives_x_device \
  --overwrite_output_dir

ckpt=

python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT -m tevatron.driver.encode \
  --output_dir=${MODEL_DIR}/${ckpt} \
  --cache_dir cache/ \
  --data_cache_dir cache/data/ \
  --model_name_or_path ${MODEL_DIR}/${ckpt} \
  --tokenizer_name ${MODEL_DIR}/passage_model \
  --fp16 \
  --per_device_eval_batch_size 1024 \
  --p_max_len 256 \
  --dataset_name json \
  --encode_in_path preprocessed/USPTO_50K/corpus.jsonl \
  --encoded_save_path ${MODEL_DIR}/${ckpt}/corpus.pkl


for split in test valid train
do
  echo $split
  python -m torch.distributed.launch --nproc_per_node=1 --master_port $MASTER_PORT -m tevatron.driver.encode \
    --output_dir=${MODEL_DIR}/${ckpt} \
    --cache_dir cache/ \
    --data_cache_dir cache/data/ \
    --model_name_or_path ${MODEL_DIR}/${ckpt} \
    --tokenizer_name ${MODEL_DIR}/query_model \
    --fp16 \
    --per_device_eval_batch_size 1024 \
    --q_max_len 128 \
    --dataset_name json \
    --encode_in_path preprocessed/USPTO_50K/${split}.jsonl \
    --encoded_save_path ${MODEL_DIR}/${ckpt}/${split}.pkl \
    --encode_is_qry

  python -m tevatron.faiss_retriever \
    --query_reps ${MODEL_DIR}/${ckpt}/${split}.pkl \
    --passage_reps ${MODEL_DIR}/${ckpt}/corpus.pkl \
    --depth 20 \
    --batch_size -1 \
    --save_json \
    --save_ranking_to ${MODEL_DIR}/${ckpt}/${split}_rank.json

done