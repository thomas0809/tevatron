import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser


def evaluate(prediction, corpus):
    hits = []
    for ex in prediction:
        if ex['id'].startswith('unk'):
            continue
        # if ex['id'] in ex['nn']:
        #     hits.append(ex['nn'].index(ex['id']))
        # else:
        #     hits.append(10000)
        gold_text = corpus.at[ex['id'], 'paragraph_text']
        hit = 10000
        for i, nn_id in enumerate(ex['nn']):
            if corpus.at[nn_id, 'paragraph_text'] == gold_text:
                hit = i
                break
        hits.append(hit)
    hits = np.array(hits)
    recall = {i: np.mean(hits < i) for i in range(1, 11)}
    return recall


def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--corpus', type=str)
    args = parser.parse_args()

    corpus = pd.read_csv(args.corpus)
    corpus.set_index("id", inplace=True)

    with open(args.file) as f:
        prediction = json.load(f)

    print(evaluate(prediction, corpus))


if __name__ == '__main__':
    main()
