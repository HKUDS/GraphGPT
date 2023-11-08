from __future__ import division
from torch.utils.data import Dataset
import numpy as np


class DataHelper(Dataset):
    def __init__(self, edge_index, args, directed=False, transform=None):
        # self.num_nodes = len(node_list)
        self.transform = transform

        self.degrees = dict()
        self.node_set = set()
        self.neighs = dict()
        self.args = args

        idx, degree = np.unique(edge_index, return_counts=True)
        for i in range(idx.shape[0]):
            self.degrees[idx[i]] = degree[i].item()

        self.node_dim = idx.shape[0]
        print("lenth of dataset", self.node_dim)

        train_edge_index = edge_index
        self.final_edge_index = train_edge_index.T

        for i in range(self.final_edge_index.shape[0]):
            s_node = self.final_edge_index[i][0].item()
            t_node = self.final_edge_index[i][1].item()

            if s_node not in self.neighs:
                self.neighs[s_node] = []
            if t_node not in self.neighs:
                self.neighs[t_node] = []

            self.neighs[s_node].append(t_node)
            if not directed:
                self.neighs[t_node].append(s_node)

        # self.neighs = sorted(self.neighs)
        self.idx = idx

        print("len of neighs", len(self.neighs))
        # print(self.neighs)

    def __len__(self):
        return self.node_dim

    def __getitem__(self, idx):
        s_n = self.idx[idx].item()
        t_n = [np.random.choice(self.neighs[s_n], replace=True).item() for _ in range(self.args.neigh_num)]
        t_n = np.array(t_n)

        sample = {
            "s_n": s_n,  # e.g., 5424
            "t_n": t_n,  # e.g., 5427
            # 'neg_n': neg_n
        }

        if self.transform:
            sample = self.transform(sample)
        # print(sample)

        return sample
