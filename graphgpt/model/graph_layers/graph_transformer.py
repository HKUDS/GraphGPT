import torch as t
from torch import nn
import torch.nn.functional as F
import math
from transformers.configuration_utils import PretrainedConfig

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = t.zeros(q_len, d_model)
    position = t.arange(0, q_len).unsqueeze(1)
    div_term = t.exp(t.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = t.sin(position * div_term)
    pe[:, 1::2] = t.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def pos_encoding(pe, learn_pe, nvar, d_model):
    # Positional encoding
    if pe == None:
        W_pos = t.empty((nvar, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = t.empty((nvar, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = t.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = t.zeros((nvar, 1))
        t.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = t.zeros((nvar, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos': W_pos = PositionalEncoding(nvar, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class graph_transformer(nn.Module):
    def __init__(self, args):
        super(graph_transformer, self).__init__()
        self.config = PretrainedConfig()
        self.gtLayers = nn.Sequential(*[GTLayer(args) for i in range(args.gt_layers)])

        self.W_pos = pos_encoding('zeros', True, 1, args.att_d_model)
                
        self.W_P = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.inverW_P = nn.Linear(args.att_d_model, args.gnn_output)
        self.args = args

    def forward(self, g):
        # Adj: sp adj
        # x: bs * n * d_model * num_patch
        
        # print(edge_index)
        device = self.parameters().__next__().device
        g = g.to(device)
        
        x = g.graph_node
        
        # x, W_P_weight, W_P_bias= Mv2Samedevice([x, self.W_P.weight, self.W_P.bias])
        # self.W_P.weight = nn.Parameter(W_P_weight.to(x.dtype))
        # self.W_P.bias = nn.Parameter(W_P_bias.to(x.dtype))
        # print(self.W_P.dtype, x.dtype)
        z = self.W_P(x)
        if self.args.if_pos: 
            embeds = self.dropout(z + self.W_pos) 
        else: 
            embeds = self.dropout(z) 
        for gt in self.gtLayers:
            embeds = gt(g, embeds) # bs * num_patch * n * d_model
        # embeds, inverW_P_weight, inverW_P_bias = Mv2Samedevice([embeds, self.inverW_P.weight, self.inverW_P.bias])
        # self.inverW_P.weight = nn.Parameter(inverW_P_weight.to(embeds.dtype))
        # self.inverW_P.bias = nn.Parameter(inverW_P_bias.to(embeds.dtype))
        ret = self.inverW_P(embeds)
        return ret
def Mv2Samedevice(vars): 
    return [var.to(vars[0].device) for var in vars]

class GTLayer(nn.Module):
    def __init__(self, args):
        super(GTLayer, self).__init__()
        self.qTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        self.kTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        self.vTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        if args.att_norm: 
            self.norm = nn.LayerNorm(args.att_d_model, eps=1e-6)
        self.args = args
        
        
    
    def forward(self, g, embeds):
        # Adj: adj
        # x: n * d_model
        rows, cols = g.edge_index
        nvar, _ = embeds.shape
        # print(rows)
        # print(cols)

        rowEmbeds = embeds[rows, :]
        colEmbeds = embeds[cols, :]
        evar, _ = rowEmbeds.shape

        # rowEmbeds, qTrans, kTrans, vTrans = Mv2Samedevice([rowEmbeds, self.qTrans, self.kTrans, self.vTrans])
        # self.qTrans = nn.Parameter(qTrans.to(rowEmbeds.dtype))
        # self.kTrans = nn.Parameter(kTrans.to(rowEmbeds.dtype))
        # self.vTrans = nn.Parameter(vTrans.to(rowEmbeds.dtype))
        qEmbeds = (rowEmbeds @ self.qTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        
        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        
        tem = t.zeros([nvar, self.args.head]).to(expAtt.device, dtype=expAtt.dtype)
        # print(tem.device, expAtt.device, rows.device)
        rows = rows.to(expAtt.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows, :]
        att = expAtt / (attNorm + 1e-8) # bleh
        
        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([evar, self.args.att_d_model])
        tem = t.zeros([nvar, self.args.att_d_model]).to(resEmbeds.device, dtype=resEmbeds.dtype)
        rows = rows.to(resEmbeds.device)
        tem = tem.to(resEmbeds.dtype)
        resEmbeds = tem.index_add_(0, rows, resEmbeds) # nd
        resEmbeds = resEmbeds + embeds
        if self.args.att_norm: 
            # resEmbeds, norm_weight, norm_bias = Mv2Samedevice([resEmbeds, self.norm.weight, self.norm.bias])
            # self.norm.weight = nn.Parameter(norm_weight.to(resEmbeds.dtype))
            # self.norm.bias = nn.Parameter(norm_bias.to(resEmbeds.dtype))
            resEmbeds = self.norm(resEmbeds)

        return resEmbeds