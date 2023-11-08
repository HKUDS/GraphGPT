import os.path as osp
from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model_gt import CLIP, tokenize
from data import DataHelper
from sklearn import preprocessing
import json
import os
from tqdm import tqdm
from utils import Logger
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F


def cal_cl_loss(s_features, t_features, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * s_features @ t_features.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def assure_dir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def main(args):
    setup_seed(seed)
    save_dir = "./res/{}/".format(args.data_name)
    logger = Logger(args, save_dir)
    model_save_name = f"{args.gnn_type}-{args.exp_time}-og.pkl"

    model = CLIP(args).to(device)
    dataset = DataHelper(arr_edge_index, args)
    model.train()

    in_g = Data(x=node_f, edge_index=edge_index).to(device)
    for j in range(args.epoch_num):
        epoch_loss = 0.0
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
        for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
            s_n, t_n = sample_batched["s_n"], sample_batched["t_n"]
            s_n_arr = s_n.numpy()  # .reshape((1, -1))
            t_n_arr = t_n.numpy().reshape(-1)
            s_n_text, t_n_text = [new_dict[i] for i in s_n_arr], [new_dict[j] for j in t_n_arr]
            s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(
                t_n_text, context_length=args.context_length
            ).to(device)

            s_n, t_n = s_n.long().to(device), t_n.long().to(device)
            s_image_features, s_text_features, t_text_features, labels = model(
                in_g, s_n, t_n, s_n_text, t_n_text, device
            )

            node_loss = cal_cl_loss(s_image_features, s_text_features, labels)
            gt_loss = cal_cl_loss(s_image_features, t_text_features, labels)
            tt_loss = cal_cl_loss(s_text_features, t_text_features, labels)

            all_loss = node_loss + args.edge_coef * gt_loss + args.edge_coef * tt_loss

            model.optim.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            model.optim.step()
            loss = round((all_loss.detach().clone()).cpu().item(), 4)

            if i_batch % 100 == 0:
                logger.log("{}th loss in {} epoch:{}".format(i_batch, j + 1, loss))
            epoch_loss += loss / len(loader)
        # break
        logger.log("{}th epoch mean loss:{}".format(j + 1, epoch_loss))
    torch.save(model.state_dict(), osp.join(save_dir, model_save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--aggregation_times", type=int, default=2, help="Aggregation times")
    parser.add_argument("--epoch_num", type=int, default=2, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--edge_coef", type=float, default=10)
    parser.add_argument("--neigh_num", type=int, default=3)

    parser.add_argument("--gnn_input", type=int, default=128)
    parser.add_argument("--gnn_hid", type=int, default=128)
    parser.add_argument("--gnn_output", type=int, default=128)

    parser.add_argument("--context_length", type=int, default=128)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49408)  # 49408
    parser.add_argument("--data_name", type=str, default="Cora")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log", type=int, default=1)

    # gt config
    parser.add_argument("--gnn_type", type=str, default="gt")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gt_layers", type=int, default=3)
    parser.add_argument("--att_d_model", type=int, default=128)
    parser.add_argument("--att_norm", type=bool, default=True)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--if_pos", type=bool, default=False)

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)

    num_nodes = 0
    tit_list = []
    tit_dict = json.load(open("./data/{}/{}_text.json".format(args.data_name, args.data_name)))
    new_dict = {}

    for i in range(len(tit_dict)):
        num_nodes += 1
        new_dict[i] = tit_dict[str(i)]

    print("num_nodes", num_nodes)

    edge_index = np.load("./data/{}/{}_edge.npy".format(args.data_name, args.data_name))

    arr_edge_index = edge_index

    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load("./data/{}/{}_f_bert.npy".format(args.data_name, args.data_name))
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(torch.float).to(device)

    start = time.perf_counter()

    seed = 1
    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
