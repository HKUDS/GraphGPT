import os
import random
from typing import Any, Optional, Dict, List
import logging
import torch
from lightning.pytorch import LightningModule
from transformers import get_linear_schedule_with_warmup, CLIPTextModel, CLIPTokenizer, PreTrainedTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from graphgpt.model.GraphLlama import GraphLlamaForCausalLM
import transformers

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class GraphGPT_pl(LightningModule): 
    def __init__(self,
        training_args, model_args, data_args, tokenizer, 
        **kwargs,
    ):
        super().__init__()
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

        bnb_model_from_pretrained_args = {}

    ## load 4 8 bit 
        if training_args.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_int8_training
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                )
            ))

        if model_args.graph_tower is not None:
            self.model = GraphLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                ) ## TODO: add real Graph Llama model 
        else:
            self.model = transformers.LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        self.model.config.pretrain_graph_model_path = self.model.config.pretrain_graph_model_path + model_args.graph_tower
        self.model.config.use_cache = False
        if model_args.freeze_backbone:
            self.model.model.requires_grad_(False)

        if training_args.bits in [4, 8]:
            self.model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
            self.model = prepare_model_for_int8_training(self.model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if training_args.gradient_checkpointing and model_args.graph_tower is None:
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            logging.warning("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        
        if model_args.graph_tower is not None:
            model_graph_dict = self.model.get_model().initialize_graph_modules(
                graph_tower=model_args.graph_tower,
                graph_select_layer=model_args.graph_select_layer,
                pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter,
                fsdp=None
            )
            self.model.get_graph_tower().to(dtype=compute_dtype)
            # graph_config = model_graph_dict['graph_config']

            # data_args.graph_token_len = model_graph_dict['graph_token_len']
            # data_args.graph_processor = model_graph_dict['graph_processor']
            data_args.is_graph = True

            self.model.config.tune_graph_mlp_adapter = training_args.tune_graph_mlp_adapter = model_args.tune_graph_mlp_adapter
            if model_args.tune_graph_mlp_adapter:
                self.model.requires_grad_(False)
                for p in self.model.get_model().graph_projector.parameters():
                    p.requires_grad = True

            self.model.config.freeze_graph_mlp_adapter = training_args.freeze_graph_mlp_adapter
            if training_args.freeze_graph_mlp_adapter:
                for p in self.model.get_model().graph_projector.parameters():
                    p.requires_grad = False

            if training_args.bits in [4, 8]:
                self.model.get_model().graph_projector.to(dtype=compute_dtype, device=training_args.device)

            self.model.config.use_graph_start_end = data_args.use_graph_start_end = model_args.use_graph_start_end
            # graph_config.use_graph_start_end = training_args.use_graph_start_end = model_args.use_graph_start_end
            training_args.use_graph_start_end = model_args.use_graph_start_end
            self.model.config.sep_graph_conv_front = data_args.sep_graph_conv_front
            self.model.initialize_graph_tokenizer(use_graph_start_end=model_args.use_graph_start_end, tokenizer=tokenizer, device='cuda',
                                            tune_graph_mlp_adapter=model_args.tune_graph_mlp_adapter, pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter)

            params_no_grad = [n for n, p in self.model.named_parameters() if not p.requires_grad]
            if training_args.bits in [4, 8]:
                from peft.tuners.lora import LoraLayer
                for name, module in self.model.named_modules():
                    if isinstance(module, LoraLayer):
                        if training_args.bf16:
                            module = module.to(torch.bfloat16)
                    if 'norm' in name:
                        module = module.to(torch.float32)
                    if 'lm_head' in name or 'embed_tokens' in name:
                        if hasattr(module, 'weight'):
                            if training_args.bf16 and module.weight.dtype == torch.float32:
                                module = module.to(torch.bfloat16)

            print('************************** parameters: #', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            tuned_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    tuned_params.append(name)
            print(tuned_params)
        
    def training_step(self, batch, batch_idx):
        bs = len(batch["input_ids"])
        loss_dict = self.model(**batch)
        loss = loss_dict['loss']
        
        log_dict = {f'train_loss': loss.item()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # no_decay = ["bias", "LayerNorm.weight"]
        # if IS_STAGE2:
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()], "lr_scale": [1e-5, 1e-4]
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.training_args.warmup_steps,
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]