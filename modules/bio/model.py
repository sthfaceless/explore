import torch.nn

from modules.common.model import *
from modules.ddd.util import *


class PointRegression(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=256, n_blocks=2, out_dims=(512, 256, 64), seq_len=400,
                 num_heads=8, dropout=0.0, agg_dim=4):
        super(PointRegression, self).__init__()
        self.seq_len = seq_len
        self.agg_dim = agg_dim

        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.embeds = torch.nn.Embedding(num_embeddings=seq_len + 1, embedding_dim=hidden_dim, padding_idx=seq_len)
        self.blocks = torch.nn.ModuleList([])
        for block_id in range(n_blocks):
            block = torch.nn.Module()
            block.norm1 = torch.nn.LayerNorm(hidden_dim)
            block.attn = Attention(input_dim=hidden_dim, embed_dim=hidden_dim, num_heads=num_heads)
            block.norm2 = torch.nn.LayerNorm(hidden_dim)
            block.dropout = torch.nn.Dropout(dropout)
            block.ff1 = torch.nn.Linear(hidden_dim, hidden_dim * 4)
            block.ff2 = torch.nn.Linear(hidden_dim * 4, hidden_dim)
            self.blocks.append(block)

        self.out_layer = torch.nn.Linear(hidden_dim, agg_dim)
        self.out_agg = torch.nn.Linear(seq_len * agg_dim, out_dims[0])

        self.out_blocks = torch.nn.ModuleList([])
        self.out_norms = torch.nn.ModuleList([])
        for prev_dim, dim in zip(out_dims, list(out_dims[1:]) + [1]):
            self.out_norms.append(norm(prev_dim))
            self.out_blocks.append(torch.nn.Linear(prev_dim, dim))

    def forward(self, seq, mask=None):

        # input mapping to internal dim
        h = self.input_layer(seq)

        # additional positional embedding
        b, seq_len, dim = seq.shape
        pos_idx = torch.arange(seq_len).type_as(seq).long().view(1, seq_len).repeat(b, 1)
        if mask is not None:
            pos_idx = torch.where(mask, pos_idx, torch.ones_like(pos_idx) * seq_len)
        h += self.embeds(pos_idx)

        # transformer encoder blocks
        for block in self.blocks:
            h = block.norm1(block.attn(h, mask=mask) + h)
            h = block.norm2(block.ff2(block.dropout2(nonlinear(block.ff1(nonlinear(h))))) + h)

        # out regression head
        h = nonlinear(self.out_layer(h).view(b, self.seq_len * self.agg_dim))
        h = self.out_agg(h)
        for gnorm, block in zip(self.out_norms, self.out_blocks):
            h = block(nonlinear(gnorm(h)))
        out = h.view(b)  # squeeze last regression dim
        return out
