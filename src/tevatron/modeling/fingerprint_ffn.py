import torch
import torch.nn as nn


class FingerprintFFN(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.ffn = nn.Linear(model_args.fp_size,
                             model_args.projection_out_dim,
                             bias=True)
        self.dropout = nn.Dropout(model_args.ffn_dropout)
        self.gelu = nn.GELU()

    def forward(self, input: torch.Tensor):
        output = self.ffn(input)
        output = self.gelu(output)
        output = self.dropout(output)

        return output
