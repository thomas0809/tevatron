import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizer


class FPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(self, tokenizer, max_p_len: int = 128):
        self.q_tokenizer, self.p_tokenizer = tokenizer
        assert self.q_tokenizer is None

        self.max_p_len = max_p_len

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = torch.stack(qq, dim=0)         # (fp_size,) -> (b, fp_size)
        d_collated = self.q_tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated
