import json
import numpy as np
from argparse import ArgumentParser


def evaluate(prediction):
    hits = []
    for ex in prediction:
        if ex['id'].startswith('unk'):
            continue
        if ex['id'] in ex['nn']:
            hits.append(ex['nn'].index(ex['id']))
        else:
            hits.append(10000)
    hits = np.array(hits)
    recall = {i: np.mean(hits < i) for i in range(1, 11)}
    return recall


def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    with open(args.file) as f:
        prediction = json.load(f)

    print(evaluate(prediction))


if __name__ == '__main__':
    main()
