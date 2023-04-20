import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

"""
python scripts/hard_negative.py --corpus data/USPTO_rxn_corpus.csv --train_file data/USPTO_50K/matched/train.csv \
    --train_pred output/rxntext_retro_ep200/train_rank.json --save_file preprocessed/USPTO_50K/train_matched_hn.jsonl
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--train_pred', type=str)
    parser.add_argument('--save_file', type=str)
    parser.add_argument('--num_neighbors', type=int, default=10)
    args = parser.parse_args()

    corpus = pd.read_csv(args.corpus, keep_default_na=False)
    corpus.set_index("id", inplace=True)

    train_df = pd.read_csv(args.train_file)
    with open(args.train_pred) as f:
        train_pred = json.load(f)

    dedup_cnt = 0
    no_neg_cnt = 0
    with open(args.save_file, "w") as of:
        for i, row in tqdm(train_df.iterrows()):
            query_id = row["matched_id"]
            if query_id.startswith('unk'):
                continue
            query = row["product_smiles"]

            instance = {
                "query_id": query_id,
                "query": query,
                "positive_passages": [{
                    "docid": query_id,
                    "title": corpus.at[query_id, "heading_text"],
                    "text": corpus.at[query_id, "paragraph_text"]
                }],
                "negative_passages": []
            }
            assert train_pred[i]['id'] == query_id
            for nn_id in train_pred[i]['nn']:
                if nn_id == query_id:
                    continue
                title = corpus.at[nn_id, 'heading_text']
                text = corpus.at[nn_id, 'paragraph_text']
                flag = False
                for neg in instance['positive_passages'] + instance['negative_passages']:
                    if neg['text'] == text:
                        flag = True
                        break
                if not flag:
                    instance['negative_passages'].append({
                        'docid': nn_id,
                        'title': title,
                        'text': text
                    })
                else:
                    dedup_cnt += 1
                if len(instance['negative_passages']) >= args.num_neighbors:
                    break
            if len(instance['negative_passages']) == 0:
                no_neg_cnt += 1
                sample_df = corpus.sample(n=args.num_neighbors)
                for sample_id, sample_row in sample_df.iterrows():
                    instance['negative_passages'].append({
                        'docid': sample_id,
                        'title': sample_row['heading_text'],
                        'text': sample_row['paragraph_text']
                    })
            of.write(f"{json.dumps(instance, ensure_ascii=False)}\n")

    print('dedup', dedup_cnt)
    print('no neg', no_neg_cnt)


if __name__ == '__main__':
    main()
