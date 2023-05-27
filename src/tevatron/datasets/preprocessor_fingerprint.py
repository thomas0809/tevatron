from ..utils.chem_utils import get_mol_fp
from typing import Union, Tuple
from transformers import PreTrainedTokenizer


class TrainPreProcessorFP:
    def __init__(self, tokenizer: Union[PreTrainedTokenizer, Tuple[PreTrainedTokenizer, PreTrainedTokenizer]],
                 model_args,
                 text_max_length=256, separator=' '):
        self.q_tokenizer, self.p_tokenizer = tokenizer
        assert self.q_tokenizer is None
        self.model_args = model_args

        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query = get_mol_fp(
            mol_smi=example["query"],
            fp_radius=self.model_args.fp_radius,
            fp_size=self.model_args.fp_size,
            dtype="int32"
        )

        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.p_tokenizer.encode(text,
                                                     add_special_tokens=False,
                                                     max_length=self.text_max_length,
                                                     truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.p_tokenizer.encode(text,
                                                     add_special_tokens=False,
                                                     max_length=self.text_max_length,
                                                     truncation=True))
        return {'query': query, 'positives': positives, 'negatives': negatives}


class QueryPreProcessorFP:
    def __init__(self, model_args):
        self.model_args = model_args

    def __call__(self, example):
        query_id = example['query_id']
        query = get_mol_fp(
            mol_smi=example["query"],
            fp_radius=self.model_args.fp_radius,
            fp_size=self.model_args.fp_size,
            dtype="int32"
        )

        return {'text_id': query_id, 'text': query}
