import json
import os
import pandas as pd
from tqdm import tqdm


def preprocess():
    data_path = "data/USPTO_50K/matched/"
    preprocessed_path = "preprocessed/USPTO_50K/"
    os.makedirs(preprocessed_path, exist_ok=True)

    corpus_file = os.path.join("data/USPTO_rxn_corpus.csv")
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
                query_id = row["matched_id"]
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
                        "title": corpus.at[query_id, "heading_text"],
                        "text": corpus.at[query_id, "paragraph_text"]
                    })
                of.write(f"{json.dumps(instance, ensure_ascii=False)}\n")


def random_negative():
    data_path = "data/USPTO_50K/matched/"
    preprocessed_path = "preprocessed/USPTO_50K/"
    os.makedirs(preprocessed_path, exist_ok=True)

    corpus_file = os.path.join("data/USPTO_rxn_corpus.csv")
    print(f"Loading corpus from {corpus_file}")
    corpus = pd.read_csv(corpus_file, keep_default_na=False)

    ofn = os.path.join(preprocessed_path, f"train_matched_rn.jsonl")
    lines = open(os.path.join(preprocessed_path, 'train_matched.jsonl')).readlines()
    with open(ofn, "w") as of:
        for line in lines:
            instance = json.loads(line)
            sample = corpus.sample(n=10)
            for i, row in sample.iterrows():
                if row['paragraph_text'] == instance['positive_passages'][0]['text']:
                    continue
                instance['negative_passages'].append({
                    'docid': row['id'],
                    'title': row['heading_text'],
                    'text': row['paragraph_text']
                })
            of.write(f"{json.dumps(instance, ensure_ascii=False)}\n")


if __name__ == "__main__":
    # preprocess()
    random_negative()
