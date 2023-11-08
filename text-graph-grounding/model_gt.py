from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Union, List
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch import nn, optim
from graph_transformer import graph_transformer

_tokenizer = _Tokenizer()


def cal_cl_loss(s_features, t_features, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * s_features @ t_features.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.context_length = args.context_length
        self.args = args
        self.edge_coef = args.edge_coef

        if args.gnn_type == "gt":
            self.gnn = graph_transformer(args)

        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = args.vocab_size
        self.token_embedding = nn.Embedding(
            args.vocab_size, args.transformer_width
        )  # the embedding for all possible tokens
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, args.transformer_width))
        self.ln_final = LayerNorm(args.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(args.transformer_width, args.embed_dim))

        if args.gnn_type == "gcn":
            self.dtype = self.gnn.vars[0].dtype
        elif args.gnn_type == "gt":
            self.dtype = self.gnn.W_pos.dtype

        self.optim = optim.Adam(
            [
                {"params": self.token_embedding.weight},
                {"params": self.positional_embedding},
                {"params": self.transformer.parameters()},
                {"params": self.text_projection},
                {"params": self.gnn.parameters()},
            ],
            lr=args.lr,
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_graph(self, idx_train, g):
        embs = self.gnn(g)
        idx_train = idx_train.to(embs.device)
        idx_train = idx_train
        train_embs = embs[idx_train]
        return train_embs

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(
            1, 0, 2
        )  # NLD -> LND, batch_size * context_length *emb_dim -> context_length * batch_size  *emb_dim
        x = self.transformer(x)
        x = x.permute(
            1, 0, 2
        )  # LND -> NLD, context_length * batch_size *emb_dim -> batch_size * context_length *emb_dim
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot （end of token） embedding (eot_token is the highest number in each sequence)
        # so there is node need to shorten the context length
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  #
        x = x @ self.text_projection
        return x

    def forward(self, g, s_n, t_n, s_n_text, t_n_text, device):
        s_image_features = self.encode_graph(s_n, g)

        s_text_features = self.encode_text(s_n_text)

        t_text_features = self.encode_text(t_n_text)
        t_text_features = t_text_features.reshape(s_image_features.shape[0], self.args.neigh_num, self.args.gnn_output)
        t_text_features = torch.mean(t_text_features, dim=1, keepdim=False)
        # normalized features
        s_image_features = s_image_features / s_image_features.norm(dim=-1, keepdim=True)
        s_text_features = s_text_features / s_text_features.norm(dim=-1, keepdim=True)
        t_text_features = t_text_features / t_text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits

        labels = torch.arange(s_image_features.shape[0]).to(device)

        return s_image_features, s_text_features, t_text_features, labels


def tokenize(texts: Union[str, List[str]], context_length: int = 128, truncate: bool = True) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result
