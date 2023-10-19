import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from graphgpt.conversation import conv_templates, SeparatorStyle
from graphgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from graphgpt.model import *
from graphgpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy
import re

import os
import os.path as osp
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json

import ray

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def load_graph(instruct_item, graph_data_path): 
    graph_data_all = torch.load(graph_data_path)
    graph_dict = instruct_item['graph']
    graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    graph_node_list = copy.deepcopy(graph_dict['node_list'])
    target_node = copy.deepcopy(graph_dict['node_idx'])
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    graph_node_rep = graph_data_all[graph_type].x[graph_node_list] ## 
    
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size

    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

    return {
        'graph_data': graph_ret, 
        'graph_token_len': cur_token_len
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')
    
    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)

    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split
            )
        )

    ans_jsons = []

    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_id, end_id):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model

    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    
    print(f'total: {len(prompt_file)}')
    res_file = osp.join(args.output_res_path, f'arxiv_test_res_{start_id}_{end_id}.json')
    if osp.exists(res_file): 
        with open(res_file, 'r') as f:
            res_data = json.load(f)
        ready_len = len(res_data)
        if ready_len == (end_id - start_id): 
            return res_data
    else: 
        res_data = []
        ready_len = 0
        print('*'*10, 'create res file', '*'*10)
        with open(res_file, 'w') as f:
            json.dump(res_data, f)
            
        

    for idx, instruct_item in tqdm(enumerate(prompt_file[ready_len:])):
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
        #     break

        qs = instruct_item["conversations"][0]["value"]
        
        pattern = r'<graph>'

        qs = re.sub(pattern, '', qs)

        conv_mode = "vicuna_v1_1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids

        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        res_data.append({"id": instruct_item["id"], "node_idx": instruct_item["graph"]["node_idx"], "res": outputs}.copy())
        with open(res_file, 'w') as f:
            json.dump(res_data, f)
    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=2)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)

    args = parser.parse_args()

    # eval_model(args)

    ray.init()
    run_eval(args, args.num_gpus)


# protobuf             4.22.3