import torch
from torch import nn
import math
import numpy as np
from PositionalEncoder import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self,
                num_tokens,
                dim_model,
                num_heads,
                num_encoder_layers,
                n_hid,
                dropout_percent,
                ) -> None:
        super().__init__()
        
        self.src_mask = None    
        
        self.model_type = "Transformer"
        self.dim_model = dim_model
        
        self.positional_encoder = PositionalEncoding(dim_model=dim_model)
        
        self.embedding = nn.Embedding(num_tokens, dim_model)
        
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=n_hid,
            num_encoder_layers= num_encoder_layers,
            dropout= dropout_percent
        )
        self.out = nn.Linear(dim_model, num_tokens)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        output = self.transformer.encoder(src, mask=self.src_mask)
        return self.out(output)