import json
import os
import random
import pandas as pd
from tqdm import tqdm

data_path = "data/USPTO_50K/matched1/"
preprocessed_path = "preprocessed/USPTO_50K/"
corpus_file = "data/USPTO_rxn_corpus_dedup.csv"
nn_file = "retrieve/output/USPTO_50K/train.json"

# data_path = "data/USPTO_50K_year/"
# preprocessed_path = "preprocessed/USPTO_50K_year/"
# corpus_file = "data/USPTO_50K_year/corpus_before_2012_dedup.csv"

def preprocess_corpus():
    print(f"Loading corpus from {corpus_file}")
    corpus = pd.read_csv(corpus_file, keep_default_na=False)

    ofn = os.path.join(preprocessed_path, 'corpus.jsonl')
    with open(ofn, "w") as of:
        for i, row in corpus.iterrows():
            instance = {
                'docid': row['id'],
                'title': row['heading_text'],
                'text': row['paragraph_text']
            }
            of.write(f"{json.dumps(instance, ensure_ascii=False)}\n")


def preprocess():
    os.makedirs(preprocessed_path, exist_ok=True)

    corpus_file = "data/USPTO_rxn_corpus_dedup.csv"
    print(f"Loading corpus from {corpus_file}")
    corpus = pd.read_csv(corpus_file, keep_default_na=False)

    corpus.set_index("id", inplace=True)

    for phase in ["train_matched", "train", "valid", "test"]:
        print(f"Matching SMILES with texts in corpus for: {phase}")
        if phase == 'train_matched':
            rxn_file = os.path.join(data_path, f"train.csv")
        else:
            rxn_file = os.path.join(data_path, f"{phase}.csv")
        rxn_df = pd.read_csv(rxn_file)

        ofn = os.path.join(preprocessed_path, f"{phase}.jsonl")
        with open(ofn, "w") as of:
            for i, row in tqdm(rxn_df.iterrows()):
                query_id = row["id"]
                corpus_id = row["corpus_id"]
                if phase == 'train_matched' and query_id.startswith('unk'):
                    continue
                query = row["product_smiles"]

                instance = {
                    "query_id": query_id,
                    "query": query,
                    "positive_passages": [],
                    "negative_passages": []
                }
                if not query_id.startswith('unk'):
                    instance['positive_passages'].append({
                        "docid": query_id,
                        "title": corpus.at[corpus_id, "heading_text"],
                        "text": corpus.at[corpus_id, "paragraph_text"]
                    })
                of.write(f"{json.dumps(instance, ensure_ascii=False)}\n")


def random_negative():
    print(f"Loading corpus from {corpus_file}")
    corpus = pd.read_csv(corpus_file, keep_default_na=False)

    ofn = os.path.join(preprocessed_path, f"train_matched_rn.jsonl")
    lines = open(os.path.join(preprocessed_path, 'train_matched.jsonl')).readlines()
    with open(ofn, "w") as of:
        for line in tqdm(lines):
            instance = json.loads(line)
            for j in range(10):
                row = corpus.iloc[random.randrange(len(corpus))]
                if row['paragraph_text'] == instance['positive_passages'][0]['text']:
                    continue
                instance['negative_passages'].append({
                    'docid': row['id'],
                    'title': row['heading_text'],
                    'text': row['paragraph_text']
                })
            of.write(f"{json.dumps(instance, ensure_ascii=False)}\n")


def similar_positive():
    corpus_file = "data/USPTO_rxn_corpus.csv"
    print(f"Loading corpus from {corpus_file}")
    corpus = pd.read_csv(corpus_file, keep_default_na=False)
    corpus.set_index("id", inplace=True)

    with open(nn_file) as f:
        neighbors = json.load(f)
        neighbors = {ex['id']: ex for ex in neighbors}

    ofn = os.path.join(preprocessed_path, f"train_matched_sprn.jsonl")
    lines = open(os.path.join(preprocessed_path, 'train_matched_rn.jsonl')).readlines()
    with open(ofn, "w") as of:
        for i, line in tqdm(enumerate(lines)):
            instance = json.loads(line)
            # assert instance['query_id'] == neighbors[i]['id']
            for nn_id in neighbors[instance['query_id']]['nn']:
                if nn_id != instance['query_id'] and not nn_id.startswith('unk'):
                    instance['positive_passages'].append({
                        'docid': nn_id,
                        'title': corpus.at[nn_id, 'heading_text'],
                        'text': corpus.at[nn_id, 'paragraph_text']
                    })
                    break
            of.write(f"{json.dumps(instance, ensure_ascii=False)}\n")


if __name__ == "__main__":
    # preprocess()
    # random_negative()
    # preprocess_corpus()
    similar_positive()
