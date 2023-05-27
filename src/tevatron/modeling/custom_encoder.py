import json
import logging
import os
import torch
from .dense import DenseModel
from .fingerprint_ffn import FingerprintFFN
from .graph_dmpnn import GraphDMPNN
from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


class CustomModel(DenseModel):
    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(qry)
        q_reps = qry_out

        return q_reps

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        if model_args.custom_model_name.startswith("fingerprint"):
            lm_q = FingerprintFFN(model_args)
        elif model_args.custom_model_name.startswith("graph"):
            lm_q = GraphDMPNN(model_args)
        else:
            raise NotImplementedError

        # load local
        if model_args.p_model_name_or_path:
            lm_p = cls.TRANSFORMER_CLS.from_pretrained(model_args.p_model_name_or_path, **hf_kwargs)
            model_args.untie_encoder = True
        else:
            raise NotImplementedError

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder
        )
        return model

    @classmethod
    def load(
            cls,
            model_args: ModelArguments,
            model_name_or_path,
            **hf_kwargs,
    ):
        if model_args.custom_model_name.startswith("fingerprint"):
            lm_q = FingerprintFFN(model_args)
        elif model_args.custom_model_name.startswith("graph"):
            lm_q = GraphDMPNN(model_args)
        else:
            raise NotImplementedError

        # load local
        _qry_model_path = os.path.join(model_name_or_path, 'query_model', "custom_model.pt")
        logger.info(f'loading query model weight from {_qry_model_path}')
        state_dict = torch.load(_qry_model_path, map_location=torch.device("cpu"))
        lm_q.load_state_dict(state_dict)

        _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
        logger.info(f'loading passage model weight from {_psg_model_path}')
        lm_p = cls.TRANSFORMER_CLS.from_pretrained(
            _psg_model_path,
            **hf_kwargs
        )

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=True
        )
        return model

    def save(self, output_dir: str):
        assert self.untie_encoder

        os.makedirs(os.path.join(output_dir, 'query_model'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'passage_model'), exist_ok=True)
        # self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
        self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))

        q_save_dir = os.path.join(output_dir, "query_model", f"custom_model.pt")
        torch.save(self.lm_q.state_dict(), q_save_dir)

        if self.pooler:
            self.pooler.save_pooler(output_dir)
