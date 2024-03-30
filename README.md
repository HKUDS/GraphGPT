# <center><img src="images/graphgpt.png" style="width: 10%"> GraphGPT: Graph Instruction Tuning for Large Language Models</center>


[Jiabin Tang](https://tjb-tech.github.io/), [Yuhao Yang](http://yuh-yang.github.io), [Wei Wei](#), [Lei Shi](#), [Lixin Su](#), [Suqi Cheng](#), [Dawei Yin](https://www.yindawei.com/) and [Chao Huang](https://sites.google.com/view/chaoh/home)*.
(*Correspondence )

**[Data Intelligence Lab](https://sites.google.com/view/chaoh/home)@[University of Hong Kong](https://www.hku.hk/)**, Baidu Inc.

-----

<a href='https://graphgpt.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='#'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<a href='https://arxiv.org/abs/2310.13023'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](#)
 • 🌐 <a href="https://mp.weixin.qq.com/s/rvKTFdCk719Q6hT09Caglw" target="_blank">中文博客</a>


This repository hosts the code, data and model weight of **GraphGPT** (SIGIR'24 full paper track).

-----------

## 🎉 News 
- [x] [2024.03.26]🎯🎯📢📢Our GraphGPT is accepted by SIGIR'24 in the Full paper track (20.1% acceptance rate)! Congrats to all GraphGPT team! 🎉🎉🎉
- [x] [2023.12.26]🎯🎯📢📢We have updated the efficient and lightweight training code. With the updated script, it is possible to perform two-stage instruction tuning on two Nvidia 3090 GPUs (24 GB each). The specific deployment and fine-tuning methods are as follows: 🎄🎄

#### 0. Environment Update: 

The lightweight training requires PyTorch 2.1+, so we need to update corresponding libraries: 

```shell
# if you have set up the env for GraphGPT earlier
pip uninstall torch
pip uninstall torchvision
pip uninstall torchaudio
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# update pyg for the PyTorch 2.1+
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# install lightning
pip install lightning
```

#### 1. Update the Graph Data

Due to compatibility issues, if you are using the previously released graph data, we recommend downloading and updating it according to the provided link: [updated graph data](https://huggingface.co/datasets/Jiabin99/All_pyg_graph_data).

#### 2. Run the Scripts

You can run the scripts as follow:

**Stage-1:**

```shell
cd path/to/GraphGPT
sh ./scripts/tune_script/graphgpt_stage1.sh
```

**Stage-2:**

```
cd path/to/GraphGPT
sh ./scripts/tune_script/graphgpt_stage2.sh
```

- [x] [2023.12.14]📢📢Thank you for the support from the research community. We have compiled a list of frequently asked questions (FAQs) regarding running and environment issues in the following **FAQ** list. Please take a look. Wishing everyone an early Merry Christmas!🎄🎄

<details>
<summary> <b>FQA</b> </summary>

- For 'pretrain_graph_model_path' is not defined. Please refer to issue [#7](https://github.com/HKUDS/GraphGPT/issues/7).
- If there is something wrong for you to use flash attetion, just comment the `replace_llama_attn_with_flash_attn()` in line 8 in https://github.com/HKUDS/GraphGPT/blob/main/graphgpt/train/train_mem.py. For more details, please refer to [#17](https://github.com/HKUDS/GraphGPT/issues/17)
- If you meet some error about package conflict or environment setup (especially fastchat), please refer to issue [#9](https://github.com/HKUDS/GraphGPT/issues/9) and issue [#11](https://github.com/HKUDS/GraphGPT/issues/11).
- If you meet `No module named 'graphgpt'` error, you could refer to issue [#56](https://github.com/HKUDS/GraphGPT/issues/56)

</details>


🎯🎯📢📢 We have made significant updates to the **models** and **data** used in our GraphGPT on 🤗 **Huggingface**. We highly recommend referring to the table below for further details: 

| 🤗 Huggingface Address                                        | 🎯 Description                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [huggingface.co/Jiabin99/GraphGPT-7B-mix-all](https://huggingface.co/Jiabin99/GraphGPT-7B-mix-all) | It's the checkpoint of our GraphGPT based on Vicuna-7B-v1.5 tuned on instruction data [Arxiv-PubMed-mix-NC-LP](https://huggingface.co/datasets/Jiabin99/Arxiv-PubMed-mix-NC-LP) |
| [huggingface.co/Jiabin99/Arxiv-PubMed-GraphCLIP-GT](https://huggingface.co/Jiabin99/Arxiv-PubMed-GraphCLIP-GT) | It's the checkpoint of the pre-trained graph transformer (GT) trained on Arxiv and PubMed using Text-Graph grounding. |
| [huggingface.co/datasets/Jiabin99/Arxiv-PubMed-mix-NC-LP](https://huggingface.co/datasets/Jiabin99/Arxiv-PubMed-mix-NC-LP) | This's the mixing instruction dataset with node classification (NC) and link prediction (LP) on Arxiv and PubMed. |
| [huggingface.co/datasets/Jiabin99/GraphGPT-eval-instruction](https://huggingface.co/datasets/Jiabin99/GraphGPT-eval-instruction) | We release all instruction dataset for our evaluation.       |
| [huggingface.co/datasets/Jiabin99/All_pyg_graph_data](https://huggingface.co/datasets/Jiabin99/All_pyg_graph_data) | We merge all utilized graph data.                            |
| [huggingface.co/datasets/Jiabin99/graph-matching](https://huggingface.co/datasets/Jiabin99/graph-matching) | This is the instruction data used in graph-matching stage.                            |


- [x] [2023.10.28]📢📢For the Chinese version of the explanation, please refer to this [article](https://mp.weixin.qq.com/s/rvKTFdCk719Q6hT09Caglw).

- [x] [2023.10.26]🔥🔥Release our utilized Instruction data.

- [x] [2023.10.26]🔥🔥Release checkpoints of our GraphGPT and pre-trained graph encoder.

- [x] [2023.10.23] 🚀🚀 The full paper of our GraphGPT is available at [https://arxiv.org/abs/2310.13023](https://arxiv.org/abs/2310.13023). Please check out it and give us more feedbacks! 

- [x] [2023.10.15] 🚀🚀 Release the code of GraphGPT.


## 👉 TODO 
- [ ] Exploring the potential of our GraphGPT for more graph learning tasks.
- [ ] ...

-----------




<span id='introduction'/>

## Brief Introduction 


we present the **GraphGPT** framework that aligns LLMs with graph structural knowledge with a graph instruction tuning paradigm.


- **Structural Information Encoding with Text-Graph Grounding.** To enhance the understanding of graph structural information by large language models, our framework emphasizes aligning the encoding of graph structures with the natural language space. This alignment aims to enable language models to effectively comprehend and interpret the structural elements of the graph, leveraging their inherent language understanding capabilities. To achieve this objective, we introduce a text-graph grounding paradigm that generates prompts designed to preserve the graph’s structural context for language models. This paradigm acts as a bridge, connecting the semantic understanding of textual information with the inherent structural relationships found within the graph.
- **Dual-Stage Graph Instruction Tuning.** The dual-stage graph instruction tuning paradigm proposed in this work builds upon the concept of instruction tuning, which has been recently introduced to enhance the adaptability of language models for specific domains. In this paradigm, we aim to align the language capacity of the model with the nuances of graph learning tasks, enabling the language model to generate more accurate and contextually appropriate responses for graph-structured data.
- **Chain-of-Thought (CoT) Distillation.** When faced with diverse graph data, language models may encounter new or unfamiliar patterns and structures. This distribution shift can pose challenges in generating accurate and coherent responses, especially when the number of node classes varies across different types of graph data. To address this challenge and boost accuracy in the presence of distribution shift, it is essential to equip our GraphGPT with step-by-step reasoning abilities. In this regard, we propose utilizing the Chain-of-Thought (COT) technique [47], which explicitly models the flow of thoughts and reasoning steps. By incorporating COT, our language model improves the coherence and consistency of generated text. It enables the model to follow a logical progression of ideas, enhancing its ability to understand and reason about the given graph data.


For more technical details, kindly refer to the [paper](https://arxiv.org/abs/2310.13023) and the project [website](https://graphgpt.github.io/) of our Graph. 


-----------

<span id='Usage'/>

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment Preparation'>2. Environment Preparation </a>
* <a href='#Training GraphGPT'>3. Training GraphGPT </a>
  * <a href='#Prepare Pre-trained Checkpoint'>3.1. Prepare Pre-trained Checkpoint</a>
  * <a href='#Self-Supervised Instruction Tuning'>3.2. Self-Supervised Instruction Tuning</a>
  * <a href='#Extract the Trained Projector'>3.3. Extract the Trained Projector</a>
  * <a href='#Task-Specific Instruction Tuning'>3.4. Task-Specific Instruction Tuning</a>
* <a href='#Evaluating GraphGPT'>4. Evaluating GraphGPT</a>
  * <a href='#Preparing Checkpoints and Data'>4.1. Preparing Checkpoints and Data</a>
  * <a href='#Running Evaluation'>4.2. Running Evaluation</a>

****



<span id='Code Structure'/>

### 1. Code Structure <a href='#all_catelogue'>[Back to Top]</a>

```
.
├── README.md
├── assets
│   ├── demo_narrow.gif
│   ├── screenshot_cli.png
│   ├── screenshot_gui.png
│   ├── server_arch.png
│   └── vicuna_logo.jpeg
├── format.sh
├── graphgpt
│   ├── __init__.py
│   ├── constants.py
│   ├── conversation.py
│   ├── eval
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── run_graphgpt.py
│   │   ├── run_graphgpt_LP.py
│   │   ├── run_vicuna.py
│   │   └── script
│   │       └── run_model_qa.yaml
│   ├── model
│   │   ├── GraphLlama.py
│   │   ├── __init__.py
│   │   ├── apply_delta.py
│   │   ├── apply_lora.py
│   │   ├── builder.py
│   │   ├── compression.py
│   │   ├── convert_fp16.py
│   │   ├── graph_layers
│   │   │   ├── __init__.py
│   │   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   │   ├── clip_graph.py
│   │   │   ├── graph_transformer.py
│   │   │   ├── mpnn.py
│   │   │   └── simple_tokenizer.py
│   │   ├── make_delta.py
│   │   ├── model_adapter.py
│   │   ├── model_registry.py
│   │   ├── monkey_patch_non_inplace.py
│   │   └── utils.py
│   ├── protocol
│   │   └── openai_api_protocol.py
│   ├── serve
│   │   ├── __init__.py
│   │   ├── api_provider.py
│   │   ├── bard_worker.py
│   │   ├── cacheflow_worker.py
│   │   ├── cli.py
│   │   ├── controller.py
│   │   ├── gateway
│   │   │   ├── README.md
│   │   │   └── nginx.conf
│   │   ├── gradio_block_arena_anony.py
│   │   ├── gradio_block_arena_named.py
│   │   ├── gradio_css.py
│   │   ├── gradio_patch.py
│   │   ├── gradio_web_server.py
│   │   ├── gradio_web_server_multi.py
│   │   ├── huggingface_api.py
│   │   ├── inference.py
│   │   ├── model_worker.py
│   │   ├── monitor
│   │   │   ├── basic_stats.py
│   │   │   ├── clean_battle_data.py
│   │   │   ├── elo_analysis.py
│   │   │   ├── hf_space_leaderboard_app.py
│   │   │   └── monitor.py
│   │   ├── openai_api_server.py
│   │   ├── register_worker.py
│   │   ├── test_message.py
│   │   └── test_throughput.py
│   ├── train
│   │   ├── graphchat_trainer.py
│   │   ├── llama_flash_attn_monkey_patch.py
│   │   ├── train_graph.py
│   │   ├── train_lora.py
│   │   └── train_mem.py
│   └── utils.py
├── playground
│   ├── inspect_conv.py
│   ├── test_embedding
│   │   ├── README.md
│   │   ├── test_classification.py
│   │   ├── test_semantic_search.py
│   │   └── test_sentence_similarity.py
│   └── test_openai_api
│       ├── anthropic_api.py
│       └── openai_api.py
├── pyproject.toml
├── scripts
│   ├── eval_script
│   │   └── graphgpt_eval.sh
│   ├── extract_graph_projector.py
│   ├── serving
│   │   ├── controller.yaml
│   │   └── model_worker.yaml
│   └── tune_script
│       ├── extract_projector.sh
│       ├── graphgpt_stage1.sh
│       └── graphgpt_stage2.sh
└── tests
    ├── test_openai_curl.sh
    ├── test_openai_langchain.py
    └── test_openai_sdk.py
```


<span id='Environment Preparation'/>


### 2. Environment Preparation  <a href='#all_catelogue'>[Back to Top]</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create -n graphgpt python=3.8

conda activate graphgpt

# Torch with CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
# To support vicuna base model
pip3 install "fschat[model_worker,webui]"
# To install pyg and pyg-relevant packages
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
# Clone our GraphGPT
git clone https://github.com/HKUDS/GraphGPT.git
cd GraphGPT
# Install required libraries
pip install -r requirements.txt
```

<span id='Training GraphGPT'/>

### 3. Training GraphGPT <a href='#all_catelogue'>[Back to Top]</a>

GraphGPT tuning paradigm consists of two stages: (1) self-supervised instruction tuning; (2) task-specific instruction tuning.

<span id='Prepare Pre-trained Checkpoint'/>

#### 3.1. Preparing Pre-trained Checkpoint  <a href='#all_catelogue'>[Back to Top]</a>
GraphGPT is trained based on following excellent existing models.
Please follow the instructions to prepare the checkpoints.

- `Vicuna`:
  Prepare our base model Vicuna, which is an instruction-tuned chatbot and base model in our implementation. Please download its weights [here](https://github.com/lm-sys/FastChat#model-weights). We generally utilize v1.1 and v1.5 model with 7B parameters.

- `Graph Encoder`:
  is used to encode graph structures. We employ text-graph grounding approach to obtain the pre-trained graph transformer model, which you could download by [graph transformer](https://huggingface.co/Jiabin99/Arxiv-PubMed-GraphCLIP-GT) and put it at [[./GraphGPT]](./GraphGPT). We also provide source codes and example Cora data for text-graph grounding at [[./text-graph-grounding]](./text-graph-grounding) for your reference.

- `Graph Data`:
  is a combination of all utilized pyg graph data that contain node features, edge_index and so on. You can download by [all_graph_data.pt](https://huggingface.co/datasets/Jiabin99/All_pyg_graph_data) and put it at [[./GraphGPT/graph_data]](./GraphGPT/graph_data)

<span id='Self-Supervised Instruction Tuning'/>

#### 3.2. Self-Supervised Instruction Tuning  <a href='#all_catelogue'>[Back to Top]</a>

* **Prepare data:** Please download our instruction tuning data [graph_matching.json](https://huggingface.co/datasets/Jiabin99/graph-matching) for the graph matching task.

* **Start tuning:** After the aforementioned steps, you could start the first stage tuning by filling blanks at [graphgpt_stage1.sh](scripts/tune_script/graphgpt_stage1.sh). There is an example as below: 

```shell
# to fill in the following path to run the first stage of our GraphGPT!
model_path=../vicuna-7b-v1.5-16k
instruct_ds=./data/stage_1/graph_matching.json
graph_data_path=./graph_data/all_graph_data.pt
pretra_gnn=clip_gt_arxiv
output_model=./checkpoints/stage_1

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
    graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

<span id='Extract the Trained Projector'/>

#### 3.3. Extract the Trained Projector  <a href='#all_catelogue'>[Back to Top]</a>

We could extract the trained projector in the stage 1 by filling blanks at [extract_projector.sh](scripts/tune_script/extract_projector.sh). There is an example as below: 

```shell
# to fill in the following path to extract projector for the first tuning stage!
src_model=./checkpoints/stage_1
output_proj=./checkpoints/stage_1_projector/stage_1_projector.bin

python3.8 ./scripts/extract_graph_projector.py \
  --model_name_or_path ${src_model} \
  --output ${output_proj}
```

<span id='Task-Specific Instruction Tuning'/>

#### 3.4. Task-Specific Instruction Tuning  <a href='#all_catelogue'>[Back to Top]</a>

* **Prepare data:** The choices of our task-specific instruction data could be diverse, e.g., standard or COT (Chain-of-Thought) node classification, link prediction or mixing data for multitasking. Please refer to the  [task_specific](https://huggingface.co/datasets/Jiabin99/Arxiv-PubMed-mix-NC-LP).

* **Start tuning:** After the aforementioned steps, you could start the second stage tuning by filling blanks at [graphgpt_stage2.sh](scripts/tune_script/graphgpt_stage2.sh). There is an example as below: 

```shell
# to fill in the following path to run the second stage of our GraphGPT!
model_path=../vicuna-7b-v1.5-16k
instruct_ds=./data/stage_2/data_all_mix.json
graph_data_path=./graph_data/all_graph_data.pt
pretra_gnn=clip_gt_arxiv
tuned_proj=./checkpoints/stage_1_projector/stage_1_projector.bin
output_model=./checkpoints/stage_2

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
    graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --pretrain_graph_mlp_adapter ${tuned_proj} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True\
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

```



<span id='Evaluating GraphGPT'/>

## 4. Evaluating GraphGPT  <a href='#all_catelogue'>[Back to Top]</a>

<span id='Preparing Checkpoints and Data'/>


#### 4.1. Preparing Checkpoints and Data <a href='#all_catelogue'>[Back to Top]</a>

* **Checkpoints:** You could try to evaluate GraphGPT by using your own model or our released checkpoints.
* **Data:** We split test sets for different graph datasets and make the instruction data for evaluation. Please refer to the  [evaluating](https://huggingface.co/datasets/Jiabin99/GraphGPT-eval-instruction).

<span id='Running Evaluation'/>

#### 4.2. Running Evaluation <a href='#all_catelogue'>[Back to Top]</a>

You could start the second stage tuning by filling blanks at [graphgpt_eval.sh](scripts/eval_script/graphgpt_eval.sh). There is an example as below: 
```shell
# to fill in the following path to extract projector for the second tuning stage!
output_model=./checkpoints/stage_2
datapath=./data/eval/arxiv_nc.json
graph_data_path=./graph_data/all_graph_data.pt
res_path=./output_stage_2_arxiv_nc
start_id=0
end_id=20000
num_gpus=2

python3.8 ./graphgpt/eval/run_graphgpt.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
```
---------


## Contact

For any questions or feedback, feel free to contact [Jiabin Tang](mailto:jiabintang77@gmail.com).


## Citation

If you find GraphGPT useful in your research or applications, please kindly cite:
```tex
@articles{tang2023graphgpt,
title={GraphGPT: Graph Instruction Tuning for Large Language Models}, 
author={Jiabin Tang and Yuhao Yang and Wei Wei and Lei Shi and Lixin Su and Suqi Cheng and Dawei Yin and Chao Huang},
year={2023},
eprint={2310.13023},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```



## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat), [LLaVa](https://github.com/haotian-liu/LLaVA), We also partially draw inspirations from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). For the text-graph grounding design, we leverages implementation from [G2P2](https://github.com/WenZhihao666/G2P2). The design of our website and README.md was inspired by [NExT-GPT](https://next-gpt.github.io/). Thanks for their wonderful works.





