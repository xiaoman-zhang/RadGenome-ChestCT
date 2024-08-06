'''
Author: xm_cmic
Date: 2024-05-01 17:31:57
LastEditors: xm_cmic
LastEditTime: 2024-06-14 08:05:12
FilePath: /src-0515/grounded_train_attn.py
Description: 


'''

import os 
import csv 
import json 
import numpy as np

import torch
import transformers
import torch.nn.functional as F
from transformers import Trainer 
from transformers import GPT2Tokenizer

from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from safetensors.torch import save_file, load_file



from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

from dataset.dataset_grounded_attn import CTRATE_Dataset
from model.grounded_model_attn import Grounded_VQA_Model_Attn


@dataclass
class DataArguments:
    # grounded_report_csv,qa_csv,qa_size_csv,mask_root_dir,anatomy_mask_root_dir
    Mode: Optional[str] = field(default="Train")
    train_json: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train.json')
    
    train_image_path_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_image_path.csv')
    test_image_path_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_image_path.csv')
    
    train_regioned_report_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_region_report.csv')
    test_regioned_report_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv')
    
    train_qa_abnormality_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_vqa_abnormality.csv')
    test_qa_abnormality_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_vqa_abnormality.csv')
    
    train_qa_location_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_vqa_location.csv')
    test_qa_location_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_vqa_location.csv')
    
    train_qa_presence_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_vqa_presence.csv')
    test_qa_presence_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_vqa_presence.csv')
    
    train_qa_size_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_vqa_size.csv')
    test_qa_size_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_vqa_size.csv')
    
    train_disorders_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/subset_2000/train_case_disorders.csv')
    test_disorders_csv: Optional[str] = field(default='./RadGenome-ChestCT/dataset/radgenome_files/validation_case_disorders.csv')
    
    train_mask_root_dir: Optional[str] = field(default='./RadGenome-ChestCT/dataset/train_region_mask')
    test_mask_root_dir: Optional[str] = field(default='./RadGenome-ChestCT/dataset/valid_region_mask')
    train_anatomy_mask_root_dir: Optional[str] = field(default='./RadGenome-ChestCT/dataset/train_anatomy_mask')
    test_anatomy_mask_root_dir: Optional[str] = field(default='./RadGenome-ChestCT/dataset/valid_anatomy_mask')

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    per_device_train_batch_size: int = field(default = 2)
    per_device_eval_batch_size: int = field(default = 1)
    gradient_accumulation_steps: int = field(default = 8)
    # eval_accumulation_steps: int = field(default = 8)
    output_dir: Optional[str] = field(default="./results/baseline")
    logging_dir: Optional[str] = field(default="./logs/baseline")
    num_train_epochs: int = field(default = 2)
    save_total_limit: int = field(default = 3)
    evaluation_strategy: Optional[str] = field(default="no")
    save_strategy: Optional[str] = field(default="steps")
    save_steps: int = field(default = 100)
    logging_steps: int = field(default = 1)
    lora_rank: int = field(default = 8)
    warmup_steps: int = field(default = 500)
    weight_ratio: float = field(default = 0.03)
    weight_decay: float = field(default = 0.00)
    learning_rate: float = field(default = 2e-5)
    optim: str = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="cosine")
    gradient_checkpointing: bool = field(default=False)
    save_on_each_node: bool = field(default = True)
    vision_learnable: bool = field(default = True)


@dataclass
class ModelArguments:
    llm_max_length: int = field(default = 1024)
    text_dim: int = field(default = 4096)
    language_backbone: str = field(default = 'LLaMA3')
    vision_pretrained: Optional[str] = field(default=None)
    
def main():
    parser = transformers.HfArgumentParser((ModelArguments,DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    torch.autograd.set_detect_anomaly(True)
    
    print("Setup Model")
    if model_args.language_backbone == 'LLaMA3':
        tokenizer = AutoTokenizer.from_pretrained('./Model/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained('./Model/MMedLM2-1_8B', use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
    if model_args.language_backbone == 'LLaMA3':
        language_model = AutoModelForCausalLM.from_pretrained('./Model/Meta-Llama-3-8B-Instruct', device_map="cpu",torch_dtype=torch.bfloat16)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False, # 训练模式
            r=training_args.lora_rank, # Lora 秩
            lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1# Dropout 比例
        )
        
        language_model = get_peft_model(language_model, config)
        language_model.print_trainable_parameters()
        
        language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        language_model.enable_input_require_grads()
        language_model.config.use_cache = False
    elif model_args.language_backbone == 'MMedLM':
        language_model = AutoModelForCausalLM.from_pretrained('./Model/MMedLM2-1_8B', device_map="cpu",torch_dtype=torch.bfloat16,trust_remote_code=True)
        for param in language_model.parameters():
            param.requires_grad = False
        language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    elif model_args.language_backbone == 'MMedLM-Train':
        language_model = AutoModelForCausalLM.from_pretrained('./Model/MMedLM2-1_8B', device_map="cpu",torch_dtype=torch.bfloat16,trust_remote_code=True)
        config = LoraConfig(
                    r = training_args.lora_rank, # Lora 秩
                    lora_alpha = 32,
                    target_modules = ["wqkv"],
                    lora_dropout = 0.1,
                    bias = 'none',
                    task_type=TaskType.CAUSAL_LM
                )
        language_model = get_peft_model(language_model, config)
        language_model.print_trainable_parameters()
    
        language_model.enable_input_require_grads()
        language_model.config.use_cache = False
        
    model = Grounded_VQA_Model_Attn(tokenizer,language_model,vision_backbone='UNET',vision_pretrained='./Models/SAT_Nano.pth',vision_learnable=training_args.vision_learnable,text_dim=model_args.text_dim)

    
    print("Setup Data")
    # json_file,mask_root_dir,anatomy_mask_root_dir,tokenizer
    Train_dataset = CTRATE_Dataset(data_args.train_json,data_args.train_mask_root_dir,data_args.train_anatomy_mask_root_dir,tokenizer=tokenizer,train=True)
    
    trainer = Trainer(
                    model=model, 
                    train_dataset = Train_dataset, 
                    args = training_args
                    )
    
    trainer.train(resume_from_checkpoint=True)
    # trainer.train()
    trainer.save_state()

        
if __name__ == "__main__":
    main()