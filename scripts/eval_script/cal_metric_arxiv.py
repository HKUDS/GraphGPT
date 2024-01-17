import json
import os.path as osp
import os
import torch as th
import re
import pandas as pd
from tqdm import tqdm 
from sklearn.metrics import classification_report


data_list = []
folder = '/path/to/results'
for filename in os.listdir(folder):
    if filename.endswith('.json'): 
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            data_list.extend(data)

print(data_list[1])

graph_data = th.load('/path/to/graph_data')['arxiv']
labels = graph_data.y

def cal_map(): 
    df = pd.read_csv('/path/to/labelidx2arxivcategeory.csv')
    label_dict = {}
    for index, line in df.iterrows(): 
        lb = line['arxiv category'].split(' ')[-1]
        lb_new = 'cs.' + lb.upper()
        label_dict[lb_new] = line['label idx']
    return label_dict

class_map = cal_map()

inverse_class_map = {}
for lb, lb_id in class_map.items():
    inverse_class_map[lb_id] = lb
    

pattern = r"cs\.[A-Z]{2}"


topk = 3

correct = 0
total = len(data_list)

trues = []
preds = []

for instruct_item in tqdm(data_list): 
    nid = instruct_item['node_idx']
    gpt_res = instruct_item['res']

    matches = list(set(re.findall(pattern, gpt_res))) # pred
    sorted_matches = sorted(matches, key=lambda x: gpt_res.index(x))

    true_y = labels[nid]

    pred_y = []
    for m in sorted_matches:
        try:
            pred_y.append(class_map[m])
        except: 
            pass
    try:
        print(sorted_matches)
        preds.append(pred_y[0])
    except: 
        
        preds.append(-1)

    trues.append(true_y.item())
    
    correct = correct + 1 if true_y in pred_y[:topk] else correct

acc = correct / total

print("Accuracy:", acc)

report = classification_report(trues, preds, digits=6)

print(report)