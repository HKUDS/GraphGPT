from statistics import mean
import torch as t
from torch import nn
import math

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


def PositionalEncoding(q_len, d_model, normalize=True):
    
    pe = t.zeros(q_len, d_model)
    position = t.arange(0, q_len).unsqueeze(1)
    div_term = t.exp(t.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = t.sin(position * div_term)
    pe[:, 1::2] = t.cos(position * div_term)

    if normalize:
        pe = (pe - pe.mean()) / (pe.std() * 10)

    return pe


def pos_encoding(pe_type, learn_pe, nvar, d_model):
    "positional encoding."
    if pe_type is None:
        W_pos = t.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe_type in ["zero", "zeros"]:
        W_pos = t.empty((nvar, d_model if pe_type == "zeros" else 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe_type in ["normal", "gauss"]:
        W_pos = t.zeros((nvar, 1))
        nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe_type == "uniform":
        W_pos = t.zeros((nvar, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe_type == "sincos":
        W_pos = PositionalEncoding(nvar, d_model)
    else:
        raise ValueError(f"{pe_type} is not a valid positional encoding type.")

    return nn.Parameter(W_pos, requires_grad=learn_pe)


class GraphTransformer(nn.Module):
    def __init__(self, args):
        
        super(GraphTransformer, self).__init__()

        self.gtLayers = nn.Sequential(*[GTLayer(args) for _ in range(args.gt_layers)])
        self.W_pos = pos_encoding("zeros", True, args.num_nodes, args.att_d_model)

        self.W_P = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.inverW_P = nn.Linear(args.att_d_model, args.gnn_output)
        self.args = args

    def forward(self, g):
        
        x = g.x
        x, self.W_P.weight, self.W_P.bias, self.W_pos = move_to_same_device([x, self.W_P.weight,
                                                                           self.W_P.bias,
                                                                           self.W_pos])

        z = self.W_P(x)

        embeds = self.dropout(z + self.W_pos) if self.args.if_pos else self.dropout(z)

        for gt in self.gtLayers:
            embeds = gt(g, embeds)

        embeds, self.inverW_P.weight, self.inverW_P.bias = move_to_same_device(
            [embeds, self.inverW_P.weight, self.inverW_P.bias]
        )

        return self.inverW_P(embeds)


def move_to_same_device(vars):
    return [var.to(vars[0].device) for var in vars]


class GTLayer(nn.Module):
    def __init__(self, args):
        
        super(GTLayer, self).__init__()

        # Parameter initialization
        self.qTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        self.kTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        self.vTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))

        if args.att_norm:
            self.norm = nn.LayerNorm(args.att_d_model)

        self.args = args

    def forward(self, g, embeds):
        
        rows, cols = g.edge_index
        nvar = embeds.shape[0]

        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        rowEmbeds, self.qTrans, self.kTrans, self.vTrans = move_to_same_device(
            [rowEmbeds, self.qTrans, self.kTrans, self.vTrans]
        )

        evar = rowEmbeds.shape[0]

        # Calculate QKV representations
        qEmbeds = (rowEmbeds @ self.qTrans).view(evar,
                                                  self.args.head,
                                                  -1)
                                                  
        kEmbeds = (colEmbeds @ self.kTrans).view(evar,
                                                  self.args.head,
                                                  -1)

        vEmbeds = (colEmbeds @ self.vTrans).view(evar,
                                                  self.args.head,
                                                  -1)

        # Calculate attention scores
        att_scores = t.einsum("ehd,ehe->eh", qEmbeds, kEmbeds).clamp(-10.0, 10.0)
        
        expAttScores = t.exp(att_scores)

        attNorms = t.zeros(nvar, self.args.head).to(expAttScores.device).index_add_(0,
            rows,
            expAttScores
        )[rows]

        att_weights = expAttScores / (attNorms + 1e-8)

        resEmbeds = t.einsum("eh,eht->ht", att_weights.unsqueeze(1), vEmbeds).view(evar,
                                                                                  -1)

        embeds += resEmbeds
        
        if hasattr(self,'norm'):
            resEmbeds = move_to_same_device([resEmbeds])
            resEmbeds = self.norm(resEmbeds)

        return embeds
