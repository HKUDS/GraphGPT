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
import random

import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp

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

def load_graph_LP(instruct_item, graph_data_path): 
    graph_data_all = torch.load(graph_data_path)
    graph_dict = instruct_item['graph']
    graph_edge_index_1 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_1'])).long()
    graph_node_list_1 = copy.deepcopy(graph_dict['node_list_1'])
    target_node_1 = copy.deepcopy(graph_dict['node_idx_1'])
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    graph_node_rep_1 = graph_data_all[graph_type].x[graph_node_list_1] ## 
    
    cur_token_len_1 = len(graph_node_rep_1)   # FIXME: 14 is hardcoded patch size

    graph_edge_index_2 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_2'])).long()
    graph_node_list_2 = copy.deepcopy(graph_dict['node_list_2'])
    target_node_2 = copy.deepcopy(graph_dict['node_idx_2'])
    graph_node_rep_2 = graph_data_all[graph_type].x[graph_node_list_2] ## 
    
    cur_token_len_2 = len(graph_node_rep_2)   # FIXME: 14 is hardcoded patch 

    graph_ret = {
        'graph_1': Data(graph_node = graph_node_rep_1, edge_index=graph_edge_index_1, target_node = torch.tensor([target_node_1])), 
        'graph_2': Data(graph_node = graph_node_rep_2, edge_index=graph_edge_index_2, target_node = torch.tensor([target_node_2]))
        }

    return {
        'graph_data': graph_ret, 
        'graph_token_len_1': cur_token_len_1, 
        'graph_token_len_2': cur_token_len_2 
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    if args.is_shuffle: 
        print('shuffle the prompt file!')
        random.seed(0) # 设置随机种子
        random.shuffle(prompt_file)
    else: 
        print('Not shuffle the prompt file!')

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
def eval_model(args, prompt_file, start_idx, end_idx):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    clip_graph, args_graph= load_model_pretrained(CLIP, './clip_gt_arxiv_pub')
    graph_tower = graph_transformer(args_graph)
    graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    
    model.get_model().graph_tower = graph_tower.cuda()
    # else:
    #     print('other')
    # print(next(graph_tower.parameters()).dtype)
    graph_tower.to(device='cuda', dtype=torch.float16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len

    res_data = []
    print(f'total: {len(prompt_file)}')
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
        #     break
        task_type = instruct_item['id'].split('_')[-1]
        if task_type != 'LP':
            graph_dict = load_graph(instruct_item, args.graph_data_path)
            graph_token_len = graph_dict['graph_token_len']
            graph_data = graph_dict['graph_data']

            qs = instruct_item["conversations"][0]["value"]
            # if use_graph_start_end:
            #     qs = qs + '\n' + DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len + DEFAULT_G_END_TOKEN
            # else:
            #     qs = qs + '\n' + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len

            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)
        else: 
            graph_dict = load_graph_LP(instruct_item, args.graph_data_path)
            graph_token_len_1 = graph_dict['graph_token_len_1']
            graph_token_len_2 = graph_dict['graph_token_len_2']
            graph_data = graph_dict['graph_data']

            qs = instruct_item["conversations"][0]["value"]
            # if use_graph_start_end:
            #     qs = qs + '\n' + DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len + DEFAULT_G_END_TOKEN
            # else:
            #     qs = qs + '\n' + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len

            # replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            # replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            # qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)

            replace_token_1 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_1
            replace_token_2 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_2

            replace_token_1 = DEFAULT_G_START_TOKEN + replace_token_1 + DEFAULT_G_END_TOKEN
            replace_token_2 = DEFAULT_G_START_TOKEN + replace_token_2 + DEFAULT_G_END_TOKEN

            if DEFAULT_GRAPH_TOKEN in qs:
                first_index = qs.find(DEFAULT_GRAPH_TOKEN)
                qs = qs[:first_index] + replace_token_1 + qs[first_index+len(DEFAULT_GRAPH_TOKEN):]

                second_index = qs.find(DEFAULT_GRAPH_TOKEN)
                qs = qs[:second_index] + replace_token_2 + qs[second_index+len(DEFAULT_GRAPH_TOKEN):]

        # if "v1" in args.model_name.lower():
        #     conv_mode = "graphchat_v1"
        # else: 
        #     raise ValueError('Don\'t support this model')
        conv_mode = "graphchat_v1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        if task_type != 'LP':
            graph_data.graph_node = graph_data.graph_node.to(torch.float16)
            graph_data = graph_data.cuda()
        else: 
            graph_data['graph_1'].graph_node = graph_data['graph_1'].graph_node.to(torch.float16)
            graph_data['graph_2'].graph_node = graph_data['graph_2'].graph_node.to(torch.float16)

            graph_data['graph_1'] = graph_data['graph_1'].cuda()
            graph_data['graph_2'] = graph_data['graph_2'].cuda()

        # graph_data.edge_index = graph_data.edge_index.to(torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graph_data=graph_data,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)

        res_data.append({"id": instruct_item["id"], "node_idx_1": instruct_item["graph"]["node_idx_1"], "node_idx_2": instruct_item["graph"]["node_idx_2"], 'truth': instruct_item["conversations"][1]["value"], "res": outputs}.copy())
        with open(osp.join(args.output_res_path, 'arxiv_test_res_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
            json.dump(res_data, fout, indent=4)
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
    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)
    parser.add_argument("--is_shuffle", type=bool, default=False)

    args = parser.parse_args()

    # eval_model(args)

    ray.init()
    run_eval(args, args.num_gpus)


# protobuf             4.22.3