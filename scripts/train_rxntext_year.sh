
MODEL_DIR=output/rxntext_year

#python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
#  --output_dir ${MODEL_DIR} \
#  --train_dir preprocessed/USPTO_condition_year/train.jsonl \
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
#  --negatives_x_device


#python -m tevatron.driver.encode \
#  --output_dir=${MODEL_DIR} \
#  --model_name_or_path ${MODEL_DIR} \
#  --tokenizer_name ${MODEL_DIR}/passage_model \
#  --fp16 \
#  --per_device_eval_batch_size 256 \
#  --p_max_len 256 \
#  --dataset_name json \
#  --encode_in_path preprocessed/USPTO_condition_MIT/corpus.jsonl \
#  --encoded_save_path ${MODEL_DIR}/corpus_full.pkl


for split in test val # train
do
  echo $split
#  python -m tevatron.driver.encode \
#    --output_dir=${MODEL_DIR} \
#    --model_name_or_path ${MODEL_DIR} \
#    --tokenizer_name ${MODEL_DIR}/query_model \
#    --fp16 \
#    --per_device_eval_batch_size 256 \
#    --q_max_len 256 \
#    --dataset_name json \
#    --encode_in_path preprocessed/USPTO_condition_year/${split}.jsonl \
#    --encoded_save_path ${MODEL_DIR}/${split}.pkl \
#    --encode_is_qry

  python -m tevatron.faiss_retriever \
    --query_reps ${MODEL_DIR}/${split}.pkl \
    --passage_reps ${MODEL_DIR}/corpus_full.pkl \
    --depth 20 \
    --batch_size -1 \
    --save_json \
    --save_ranking_to ${MODEL_DIR}/${split}_rank_full.json

done