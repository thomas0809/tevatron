from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Union, Tuple
from .preprocessor_fingerprint import TrainPreProcessorFP, QueryPreProcessorFP
# from .preprocessor_graph import TrainPreProcessorG, QueryPreProcessorG
from ..arguments import DataArguments, ModelArguments


class HFTrainDataset:
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, Tuple[PreTrainedTokenizer, PreTrainedTokenizer]],
                 data_args: DataArguments,
                 model_args: ModelArguments,
                 cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files,
                                    cache_dir=cache_dir,
                                    use_auth_token=True)[data_args.dataset_split]
        self.model_args = model_args
        if model_args.custom_model_name.startswith("fingerprint"):
            self.preprocessor = TrainPreProcessorFP
        elif model_args.custom_model_name.startswith("graph"):
            self.preprocessor = TrainPreProcessorG
        else:
            raise NotImplementedError

        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = data_args.passage_field_separator

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.model_args, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running featurizer and tokenizer on train dataset",
            )
        return self.dataset


class HFQueryDataset:
    def __init__(self, data_args: DataArguments, model_args: ModelArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir, use_auth_token=True
                                    )[data_args.dataset_split]
        if model_args.custom_model_name.startswith("fingerprint"):
            self.preprocessor = QueryPreProcessorFP
        elif model_args.custom_model_name.startswith("graph"):
            self.preprocessor = QueryPreProcessorG
        else:
            raise NotImplementedError

        self.model_args = model_args
        self.proc_num = data_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(model_args=self.model_args),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running featurizer",
            )
        return self.dataset
