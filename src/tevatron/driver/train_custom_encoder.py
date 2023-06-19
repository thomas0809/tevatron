import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import HfArgumentParser, set_seed

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
# from tevatron.data import TrainDataset, QPCollator
from tevatron.data_custom_encoder import TrainDataset, FPCollator#, GPCollator
from tevatron.modeling import DenseModel, CustomModel
from tevatron.trainer import TevatronTrainer as Trainer, GCTrainer
from tevatron.datasets.dataset_custom import HFTrainDataset

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) "
            f"already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    # num_labels = 1
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     cache_dir=model_args.cache_dir,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir, use_fast=False
    # )
    tokenizer = None
    if model_args.p_model_name_or_path:
        p_tokenizer = AutoTokenizer.from_pretrained(
            model_args.p_model_name_or_path, cache_dir=model_args.cache_dir, use_fast=False)
        tokenizer = (tokenizer, p_tokenizer)
    # model = DenseModel.build(
    #     model_args,
    #     training_args,
    #     # config=config,
    #     cache_dir=model_args.cache_dir,
    # )
    model = CustomModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir
    )

    hf_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args, model_args=model_args,
                                cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = TrainDataset(data_args, hf_dataset.process(), tokenizer)
    # val_dataset = TrainDataset(data_args, hf_dataset.process('val'), tokenizer)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    if model_args.custom_model_name.startswith("fingerprint"):
        data_collator = FPCollator(tokenizer, max_p_len=data_args.p_max_len)
    elif model_args.custom_model_name.startswith("graph"):
        data_collator = GPCollator(tokenizer, max_p_len=data_args.p_max_len)
    else:
        raise NotImplementedError

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        # data_collator=QPCollator(
        #     tokenizer,
        #     max_p_len=data_args.p_max_len,
        #     max_q_len=data_args.q_max_len
        # ),
        data_collator=data_collator
    )
    train_dataset.trainer = trainer
    # val_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    # Save tokenizer
    if trainer.is_world_process_zero():
        if model_args.p_model_name_or_path:
            q_tokenizer, p_tokenizer = tokenizer
            if q_tokenizer is not None:
                q_tokenizer.save_pretrained(os.path.join(training_args.output_dir, 'query_model'))
            p_tokenizer.save_pretrained(os.path.join(training_args.output_dir, 'passage_model'))
        else:
            tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
