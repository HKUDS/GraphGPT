import json
import random
import re
import pandas as pd
from tqdm import tqdm
import torch as th
from torch_geometric.utils import subgraph
from torch_geometric.data import NeighborSampler, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.utils import from_scipy_sparse_matrix
import logging
import copy

def get_logger(fname,): 
    # 1. 获取logger对象,这是日志记录的入口
    logger = logging.getLogger('process logging')

    # 2. 设置日志级别 
    logger.setLevel(logging.INFO)

    # 3. 创建日志文件handler
    log_file = f'./log_dir/{fname}.log'
    file_handler = logging.FileHandler(log_file)

    # 4. 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter) 

    # 5. 将handler添加到logger
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler() 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 6. 记录日志  
    # logger.info('App started') 
    return logger
logger =  get_logger('process_train_arxiv')
# 设置全局种子
th.manual_seed(0)
random.seed(123)  # 设置随机数种子，这里使用了整数123作为种子
instruct_list = {}
batch_id_list = [
     [0, 27560], 
     [27560, 42005], 
     [42005, 56450],
     [56450, 81915], 
     [81915, 97408],
     [97408, 112900], 
     [112900, 137294], 
     [137294, 153319],
     [153319, 169343]
     ]
for batch_id in batch_id_list:
    s_idx, e_idx = batch_id
    with open(f'./tjb_cot_result/arxiv_cot_pred_{s_idx}_{e_idx}.json') as f:
        instruct_item = json.load(f)
        assert len(instruct_item) == (e_idx - s_idx)
        ins_key = list(range(s_idx, e_idx))
        instruct_list.update(zip(ins_key, instruct_item))
print(instruct_list[0]['instruction'])
print(instruct_list[0]['input'])
print(instruct_list[0]['output'])

tra_df = pd.read_csv('./res_df/tra_df.csv')
val_df = pd.read_csv('./res_df/val_df.csv')
tst_df = pd.read_csv('./res_df/tst_df.csv')
res_df = pd.concat([tra_df, val_df, tst_df])

print(res_df[res_df['node_idx'] == 0])

'''
instruct dataset: 
[{'id': 'dsname_train_nodeidx', 'graph': [edge_row, edge_col], 'conversations': [{'from': 'human', 'value': 'human prompting.\n<graph>'}, {'from': 'gpt', 'value': 'gpt response'}]}, {...}]

graph_token: <graph>
'''
dsname ='arxiv'
split_type = 'test'

instruct_ds = []



graph_data = th.load('./tjb_cot_result/graph_data.pt')['arxiv']
print(graph_data.test_mask)
indices = th.nonzero(graph_data.test_mask).reshape(-1)
select_idx = indices.tolist()
# select_idx = [0, 1]
print(indices)

print(graph_data.edge_index)
s = graph_data.edge_index.to_scipy() # 转换为稀疏张量

edge_index, edge_attr = from_scipy_sparse_matrix(s) # 转换为COO格式
# edge_index = th.stack([row, col], dim=0)
print(f'is undirected: {is_undirected(edge_index)}')
pyg_data = Data(edge_index = edge_index, edge_attr = edge_attr, num_nodes = graph_data.num_nodes)
# Data(num_nodes=169343, x=[169343, 128], node_year=[169343, 1], y=[169343, 1], adj_t=[169343, 169343, nnz=1166243], train_mask=[169343], val_mask=[169343], test_mask=[169343], edge_index=[169343, 169343, nnz=2315598])

def cal_acc(output_item, input_item, nidx, topk): 
    title_pattern = r'Title: (.*?) \n'
    pattern = r"cs\.[A-Z]{2}"
    title = re.search(title_pattern, input_item).group(1)
    # print(title)
    
    matches = list(set(re.findall(pattern, output_item))) # pred
    sorted_matches = sorted(matches, key=lambda x: output_item.index(x))
    # result_list[nidx] = matches
    # print(res_df[res_df['node_idx'] == nidx])
    assert res_df[res_df['node_idx'] == nidx]['title'].values[0] == title, '{} ! = {}'.format(res_df[res_df['node_idx'] == nidx]['title'].values[0], title)
    true_item = 'cs.' + res_df[res_df['node_idx'] == nidx]['label'].values[0].upper()
    # print(true_item)
    # print(sorted_matches)
    return true_item in sorted_matches[:topk]

for nidx in tqdm(select_idx): 
    center_node = nidx 
    num_hops = 2
    num_neighbors = 10

    # 邻居采样    
    sampler = NeighborLoader(pyg_data, input_nodes=th.Tensor([center_node]).long(),
                            num_neighbors=[num_neighbors] * num_hops, 
                            batch_size=1)

    # 获取子图    
    sampled_data = next(iter(sampler))
    # for sampled_data in sampler:

    try:
        if cal_acc(instruct_list[nidx]['output'], instruct_list[nidx]['instruction'], nidx, topk=2) is False: 
            temp_dict = {}
            temp_dict['id'] = f'{dsname}_{split_type}_{nidx}'
            temp_dict['graph'] = {'node_idx':nidx, 'edge_index': sampled_data.edge_index.tolist(), 'node_list': sampled_data.n_id.tolist()}
            conv_list = []
            conv_temp = {}
            conv_temp['from'] = 'human'
            conv_temp['value'] = 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n' + instruct_list[nidx]['instruction'] + instruct_list[nidx]['input']
            conv_list.append(copy.deepcopy(conv_temp))

            conv_temp['from'] = 'gpt'
            conv_temp['value'] = instruct_list[nidx]['output']
            conv_list.append(copy.deepcopy(conv_temp))

            temp_dict['conversations'] = conv_list

            instruct_ds.append(temp_dict)
    except Exception as e:
        logger.info(e)
        


logger.info(f'total item: {len(instruct_ds)}')
with open(f'./instruct_ds/{dsname}_{split_type}_instruct_new.json', 'w') as f:
    json.dump(instruct_ds, f)

