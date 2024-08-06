'''
Author: xm_cmic
Date: 2024-05-25 19:36:48
LastEditors: xm_cmic
LastEditTime: 2024-06-16 09:56:52
FilePath: /src-0515/grounded_inference_attn.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
'''
Author: xm_cmic
Date: 2024-05-03 10:28:14
LastEditors: xm_cmic
LastEditTime: 2024-05-25 19:36:48
FilePath: /src-0515/grounded_inference.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import os 
import csv 
import json 
import random
import numpy as np
import pandas as pd 
import nibabel as nib

import torch
import transformers
import torch.nn.functional as F
from transformers import Trainer 
from transformers import GPT2Tokenizer
from safetensors.torch import save_file, load_file

from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from peft import PeftModel
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model,AutoPeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

from model.grounded_model_attn import Grounded_VQA_Model_Attn


from monai.transforms import (
    Resized,
    Compose
)


@dataclass
class DataArguments:
    # grounded_report_csv,qa_csv,qa_size_csv,mask_root_dir,anatomy_mask_root_dir
    Mode: Optional[str] = field(default="Train")
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
    per_device_train_batch_size: int = field(default = 1)
    per_device_eval_batch_size: int = field(default = 1)
    gradient_accumulation_steps: int = field(default = 1)
    # eval_accumulation_steps: int = field(default = 8)
    output_dir: Optional[str] = field(default="./results/llama3_global")
    num_train_epochs: int = field(default = 10)
    save_total_limit: int = field(default = 3)
    evaluation_strategy: Optional[str] = field(default="no")
    save_strategy: Optional[str] = field(default="steps")
    task: Optional[str] = field(default="report")
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
    vision_learnable: bool = field(default=False)
    report_idx: int = field(default = 0)
    
@dataclass
class ModelArguments:
    llm_max_length: int = field(default = 1024)
    text_dim: int = field(default = 4096)
    language_backbone: str = field(default = 'LLaMA3')
    vision_pretrained: Optional[str] = field(default=None)
    
    
def find_max_checkpoint_folder(path):
    # 获取路径下所有的文件夹和文件
    entries = os.listdir(path)
    
    # 过滤出所有以 "checkpoint-" 开头的文件夹
    checkpoint_folders = [folder for folder in entries if folder.startswith("checkpoint-") and os.path.isdir(os.path.join(path, folder))]
    
    # 提取文件夹名称中的数字，并找出最大的数字
    max_num = -1
    max_folder = None
    for folder in checkpoint_folders:
        try:
            num = int(folder.split("-")[1])  # 分割字符串并转换成整数
            if num > max_num:
                max_num = num
                max_folder = folder
        except ValueError:
            # 如果转换失败，忽略这个文件夹
            continue
    
    return os.path.join(path,max_folder)


def zscore_process(image):
    '''
        zscore normalization
    '''
    lower_bound, upper_bound = -1000, 200
    image = np.clip(image, lower_bound, upper_bound)
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    image = (image - mean) / (max(std, 1e-8))
    return image


def read_nii_img(nii_path):
    try:
        img = nib.load(nii_path)
        img_array = img.get_fdata()
    except FileNotFoundError:
        print(f"Warning: File not found at {nii_path}. Returning a random array instead.")
        img_array = np.random.rand(256, 256, 64)  # Generates random floats
    except Exception as e:
        print(f"An error occurred while loading the NIfTI file: {e}")
        img_array = np.random.rand(256, 256, 64)  # Backup in case of other exceptions
    return img_array
    

if __name__ == "__main__":
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
    else:
        language_model = AutoModelForCausalLM.from_pretrained('./Model/MMedLM2-1_8B', device_map="cpu",torch_dtype=torch.bfloat16,trust_remote_code=True)
        for param in language_model.parameters():
            param.requires_grad = False
            
    print("Start Load Model")
    ckp_dir = find_max_checkpoint_folder(training_args.output_dir) 
    ckp = ckp_dir + '/model.safetensors'

    model = Grounded_VQA_Model_Attn(tokenizer,language_model,vision_backbone='UNET',vision_pretrained='./checkpoint/SAT_Nano.pth',vision_learnable=training_args.vision_learnable,text_dim=model_args.text_dim)
    checkpoint = load_file(ckp)
    model.load_state_dict(checkpoint)#,strict=False)
    model = model.to('cuda')
    model.eval()
    print("Finished Load Model")
    
    print("Load Test Data")
    report_df = pd.read_csv(data_args.test_image_path_csv)
    volume_to_nii_path_dict = dict(zip(report_df['Volumename'], report_df['nii_path']))
    
    region_report_df = pd.read_csv(data_args.test_regioned_report_csv)
    # 检查并填充NaN值
    region_report_df['Anatomy'].fillna('whole scan', inplace=True)
    # 提取region列的值
    region_report_df['Region'] = region_report_df['Anatomy'].apply(lambda x: x.split('/')[0].strip().lower())
        
        
    qa_df_with_abnormalities = pd.read_csv(data_args.test_qa_abnormality_csv)
    qa_df_with_location = pd.read_csv(data_args.test_qa_location_csv)
    qa_df_with_presence = pd.read_csv(data_args.test_qa_presence_csv)
    qa_size_df = pd.read_csv(data_args.test_qa_size_csv)
    case_disorder_df = pd.read_csv(data_args.test_disorders_csv)
    
    abnormality_template_json = './data/ctrate/generate_data/qa_templates/abnormality_template.json'
    with open(abnormality_template_json, 'r') as file:
        abnormality_template = json.load(file)['abnormality']
    
    location_template_json = './data/ctrate/generate_data/qa_templates/location_template.json'
    with open(location_template_json, 'r') as file:
        location_template = json.load(file)['location']
    
    presence_template_json = './data/ctrate/generate_data/qa_templates/presence_template.json'
    with open(presence_template_json, 'r') as file:
        presence_template = json.load(file)['presence']
    
    size_template_json = './data/ctrate/generate_data/qa_templates/size_template.json'
    with open(size_template_json, 'r') as file:
        size_template = json.load(file)['size']
        
    disorder_template_json = './data/ctrate/generate_data/qa_templates/disorders_template.json'
    with open(disorder_template_json, 'r') as file:
        disorders_template = json.load(file)['disorders']
    
    augmentator =  Resized(
                spatial_size = [256,256,64],
                keys=["image","seg"],
                mode=['area',"area"]
            )
    
    print('Start Testing')
    if training_args.task == "report":
        report_idx = training_args.report_idx
        volume_list = region_report_df['Volumename'].tolist()[report_idx*(len(region_report_df)//5):(report_idx+1)*(len(region_report_df)//5)]
        anatomy_list = region_report_df['Anatomy'].tolist()[report_idx*(len(region_report_df)//5):(report_idx+1)*(len(region_report_df)//5)]
        report_list = region_report_df['Sentence'].tolist()[report_idx*(len(region_report_df)//5):(report_idx+1)*(len(region_report_df)//5)]
        # nii_path_list = region_report_df['nii_path'].tolist()
        # Open the CSV file for writing
        file_path = os.path.join(training_args.output_dir, f'tset_results_report_{str(report_idx)}.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header
        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Report', 'nii_path', 'Output_sentence', 'GT_sentence'])
        
            for idx in tqdm(range(start_idx, len(region_report_df))):
                volume = volume_list[idx]
                
                anatomy = anatomy_list[idx]
                report = report_list[idx]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                img_array = zscore_process(img_array)
                img_array = img_array[np.newaxis,:,:]
                
                node_anatomy = anatomy.split('/')[-1]
                head_anatomy = anatomy.split('/')[0]
                if node_anatomy == "others" or node_anatomy == 'whole scan':
                    mask_array = np.ones(img_array.shape)
                else:
                    try:
                        mask_dir = os.path.join(data_args.test_anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                        # print(mask_path,anatomy,node_anatomy)
                        if not os.path.exists(mask_path):
                            region_mask_dir = os.path.join(data_args.test_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                            head_anatomy = anatomy.split('/')[0]
                            mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                            # print(region_mask_dir,anatomy,head_anatomy)
                        mask_array = read_nii_img(mask_path)
                        mask_array = mask_array[np.newaxis,:,:]
                    except:
                        mask_array = np.ones(img_array.shape)
                
                data_dict = {'image': img_array, 'seg':mask_array}
                aug_data_dict = augmentator(data_dict)
                image = aug_data_dict['image']
                seg = aug_data_dict['seg']
            
                image = image.unsqueeze(0).to('cuda')
                seg = seg.unsqueeze(0).to('cuda')
                

                query_text = f'<Grounded report generation> Please describe the detailed finding of the {node_anatomy} region.'
                input_text = f'<Grounded report generation> Please describe the detailed finding of the {node_anatomy} region. <Report> {report}'
                # query_text = f'<Grounded report generation> Please describe the detailed finding of the give region.'
                # input_text = f'<Grounded report generation> Please describe the detailed finding of the give region. <Region> The given region is {node_anatomy}. This is belong to {head_anatomy}. <Report> {report}'
                
                # input_text = f'<Grounded abnormality detection> {input_question} <Region> The given region is {node_anatomy}. This is belong to {head_anatomy}. {answer}'
 
                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (2048 - 128 - len(input_ids))  # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*128 + label + [-100] * (2048 - 128 - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image = image, input_mask = seg, input_ids=input_ids,labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, report, image_path, output_sentence, gt_sentence])
                    file.flush()
    elif training_args.task == "abnormality":
        volume_list = qa_df_with_abnormalities['Volumename'].tolist()
        anatomy_list = qa_df_with_abnormalities['Anatomy'].tolist()
        abnormality_list = qa_df_with_abnormalities['Abnormality'].tolist()
        # Open the CSV file for writing
        
        file_path = os.path.join(training_args.output_dir, 'tset_results_abnormality.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header
        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Abnormality', 'nii_path', 'Output_result', 'GT_result'])
        
            for idx in tqdm(range(start_idx, len(qa_df_with_abnormalities))):
                volume = volume_list[idx]
                anatomy = anatomy_list[idx]
                abnormality = abnormality_list[idx]
                node_anatomy = anatomy.split('/')[-1]
                
                is_nan = pd.isna(abnormality)
                if is_nan:
                    abnormality = 'No Findings'
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                img_array = zscore_process(img_array)
                img_array = img_array[np.newaxis,:,:]
                
                if node_anatomy == "others":
                    mask_array = np.ones(img_array.shape)
                else:
                    try:
                        mask_dir = os.path.join(data_args.test_anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                        # print(mask_path,anatomy,node_anatomy)
                        if not os.path.exists(mask_path):
                            region_mask_dir = os.path.join(data_args.test_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                            head_anatomy = anatomy.split('/')[0]
                            mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                            # print(region_mask_dir,anatomy,head_anatomy)
                        mask_array = read_nii_img(mask_path)
                        mask_array = mask_array[np.newaxis,:,:]
                    except:
                        mask_array = np.ones(img_array.shape)
                
                data_dict = {'image': img_array, 'seg':mask_array}
                aug_data_dict = augmentator(data_dict)
                image = aug_data_dict['image']
                seg = aug_data_dict['seg']
                
                image = image.unsqueeze(0).to('cuda')
                seg = seg.unsqueeze(0).to('cuda')
                
                query_text = f'<Grounded abnormality detection>  What are the abnormalities in the {node_anatomy}?'
                input_text = f'<Grounded abnormality detection>  What are the abnormalities in the {node_anatomy}? <Abnormality> {abnormality}'# <|end_of_text|>'

                # query_text = f'<Grounded report generation> Please describe the detailed finding of the give region.'
                # input_text = f'<Grounded report generation> Please describe the detailed finding of the give region. <Region> The given region is {node_anatomy}. This is belong to {head_anatomy}. <Abnormality> {abnormality}'# <|end_of_text|>'
                
                
                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (2048 - 128 - len(input_ids))  # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*128 + label + [-100] * (2048 - 128 - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image = image, input_mask = seg, input_ids=input_ids,labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, abnormality, image_path, output_sentence, gt_sentence])
                    file.flush()
    elif training_args.task == "presence":
        abnormality_volume_list = qa_df_with_presence['Volumename'].tolist()
        abnormality_anatomy_list = qa_df_with_presence['Anatomy'].tolist()
        abnormality_abnormality_list = qa_df_with_presence['Finding'].tolist()
        abnormality_answer_list = qa_df_with_presence['Presence'].tolist()
        
        file_path = os.path.join(training_args.output_dir, 'tset_results_presence.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header
        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Abnormality', 'nii_path', 'Output_result', 'GT_result'])
        
            for idx in tqdm(range(start_idx, len(abnormality_volume_list))):
                volume = abnormality_volume_list[idx]
                anatomy = abnormality_anatomy_list[idx]
                abnormality = abnormality_abnormality_list[idx]
                answer = abnormality_answer_list[idx]
                node_anatomy = anatomy.split('/')[-1]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                img_array = zscore_process(img_array)
                img_array = img_array[np.newaxis,:,:]
                
                
                if node_anatomy == "others":
                    mask_array = np.ones(img_array.shape)
                else:
                    try:
                        mask_dir = os.path.join(data_args.test_anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                        # print(mask_path,anatomy,node_anatomy)
                        if not os.path.exists(mask_path):
                            region_mask_dir = os.path.join(data_args.test_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                            head_anatomy = anatomy.split('/')[0]
                            mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                            # print(region_mask_dir,anatomy,head_anatomy)
                        mask_array = read_nii_img(mask_path)
                        mask_array = mask_array[np.newaxis,:,:]
                    except:
                        mask_array = np.ones(img_array.shape)
                
                data_dict = {'image': img_array, 'seg':mask_array}
                aug_data_dict = augmentator(data_dict)
                image = aug_data_dict['image']
                seg = aug_data_dict['seg']
                
                image = image.unsqueeze(0).to('cuda')
                seg = seg.unsqueeze(0).to('cuda')
                
                query_text = f'<Grounded abnormality presence>  Is there any sign of {abnormality} in the {node_anatomy}?'
                input_text = f'<Grounded abnormality presence>  Is there any sign of {abnormality} in the {node_anatomy}? <Presence> {answer}'# <|end_of_text|>'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (2048 - 128 - len(input_ids))  # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*128 + label + [-100] * (2048 - 128 - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image = image,  input_mask = seg, input_ids=input_ids,labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, abnormality, image_path, output_sentence, gt_sentence])
                    file.flush()
    elif training_args.task == "location":
        abnormality_volume_list = qa_df_with_location['Volumename'].tolist()
        abnormality_anatomy_list = qa_df_with_location['Anatomy'].tolist()
        abnormality_abnormality_list = qa_df_with_location['Abnormality'].tolist()
        
        file_path = os.path.join(training_args.output_dir, 'tset_results_location.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header
        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Abnormality', 'nii_path', 'Output_result', 'GT_result'])
        
            for idx in tqdm(range(start_idx, len(qa_df_with_location))):
                volume = abnormality_volume_list[idx]
                anatomy = abnormality_anatomy_list[idx]
                abnormality = abnormality_abnormality_list[idx]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                img_array = zscore_process(img_array)
                img_array = img_array[np.newaxis,:,:]
                mask_array = np.ones(img_array.shape)
                
                data_dict = {'image': img_array, 'seg':mask_array}
                aug_data_dict = augmentator(data_dict)
                image = aug_data_dict['image']
                seg = aug_data_dict['seg']
                
                image = image.unsqueeze(0).to('cuda')
                seg = seg.unsqueeze(0).to('cuda')
                
                
                query_text = f'<Abnormality location>  Where is the {abnormality} located in the image?'
                input_text = f'<Abnormality location>  Where is the {abnormality} located in the image? <Location> {anatomy.capitalize()}.'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (2048 - 128 - len(input_ids))  # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*128 + label + [-100] * (2048 - 128 - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                 
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image = image, input_mask = seg,  input_ids=input_ids,labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, abnormality, image_path, output_sentence, gt_sentence])
                    file.flush()
    elif training_args.task == "size":
        volume_list = qa_size_df['Volumename'].tolist()
        anatomy_list = qa_size_df['Anatomy'].tolist()
        abnormality_list = qa_size_df['Abnormality'].tolist()
        size_list = qa_size_df['Size'].tolist()
        
        file_path = os.path.join(training_args.output_dir, 'tset_results_size.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header
        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'Anatomy', 'Abnormality', 'nii_path', 'Output_result', 'GT_result'])
            
            for idx in tqdm(range(start_idx, len(qa_size_df))):
                volume = volume_list[idx]
                anatomy = anatomy_list[idx]
                abnormality = abnormality_list[idx]
                size = size_list[idx]
                node_anatomy = anatomy.split('/')[-1]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                img_array = zscore_process(img_array)
                img_array = img_array[np.newaxis,:,:]
                
                if node_anatomy == "others":
                    mask_array = np.ones(img_array.shape)
                else:
                    try:
                        mask_dir = os.path.join(data_args.test_anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                        # print(mask_path,anatomy,node_anatomy)
                        if not os.path.exists(mask_path):
                            region_mask_dir = os.path.join(data_args.test_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                            head_anatomy = anatomy.split('/')[0]
                            mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                            # print(region_mask_dir,anatomy,head_anatomy)
                        mask_array = read_nii_img(mask_path)
                        mask_array = mask_array[np.newaxis,:,:]
                    except:
                        mask_array = np.ones(img_array.shape)
                
                data_dict = {'image': img_array, 'seg':mask_array}
                aug_data_dict = augmentator(data_dict)
                image = aug_data_dict['image']
                seg = aug_data_dict['seg']
                
                image = image.unsqueeze(0).to('cuda')
                seg = seg.unsqueeze(0).to('cuda')
                
                
                node_anatomy = anatomy.split('/')[-1]
                query_text = f'<Abnormality size prediction>  What is the approximate size of the {abnormality} in the {anatomy}?'
                input_text = f'<Abnormality size prediction>  What is the approximate size of the {abnormality} in the {anatomy}? <Size> {size}.'# <|end_of_text|>'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (2048 - 128 - len(input_ids))  # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*128 + label + [-100] * (2048 - 128 - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image = image,input_mask = seg, input_ids=input_ids,labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, anatomy, abnormality, image_path, output_sentence, gt_sentence])
                    file.flush()
    elif training_args.task == "disorder":
        volume_list = case_disorder_df['Volumename'].tolist()
        disorders_list = case_disorder_df['Disorders'].tolist()
        # Open the CSV file for writing
        file_path = os.path.join(training_args.output_dir, 'tset_results_disorder.csv')

        if os.path.exists(file_path):
            # Read the existing file to determine where to start appending
            with open(file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_data = list(reader)
                start_idx = len(existing_data) - 1  # Assuming the first row is the header
        else:
            start_idx = 0  # Start from the beginning if the file does not exist


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if start_idx == 0:
                # Write the header if the file is new
                writer.writerow(['Volumename', 'nii_path', 'Output_result', 'GT_result'])
            
            for idx in tqdm(range(start_idx, len(case_disorder_df))):
                volume = volume_list[idx]
                disorders = disorders_list[idx]
                
                # process_image 
                image_path = volume_to_nii_path_dict[volume]
                img_array = read_nii_img(image_path)
                img_array = zscore_process(img_array)
                img_array = img_array[np.newaxis,:,:]
                mask_array = np.ones(img_array.shape)
                
                data_dict = {'image': img_array, 'seg':mask_array}
                aug_data_dict = augmentator(data_dict)
                image = aug_data_dict['image']
                seg = aug_data_dict['seg']
                
                image = image.unsqueeze(0).to('cuda')
                seg = seg.unsqueeze(0).to('cuda')
                
                query_text = f'<Disorder prediction> What are the abnormalities in the scan?'
                input_text = f'<Disorder prediction> What are the abnormalities in the scan? <Disorder> {disorders.capitalize()}'# <|end_of_text|>'

                # Tokenize input text without adding special tokens
                tokenizer_output = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                input_ids = tokenizer_output['input_ids']

                # Copy input_ids to label and set eos_token_id locations to -100
                label = input_ids.clone()

                # Process query text if provided
                if query_text:
                    query_tokenizer_output = tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
                    query_input_ids = query_tokenizer_output['input_ids']
                    # Set the labels for the length of the query input ids to -100
                    label[:, :query_input_ids.size(1)] = -100

                # Convert input_ids to list and append eos_token_id, then pad to 960 (1024 - 64) as per your requirement
                input_ids = input_ids.squeeze().tolist() + [tokenizer.eos_token_id]
                input_ids = input_ids + [tokenizer.pad_token_id] * (2048 - 128 - len(input_ids))  # Padding input_ids

                # Convert label to list, append eos_token_id, and pad both to 1024
                label = label.squeeze().tolist() + [tokenizer.eos_token_id]
                label = [-100]*128 + label + [-100] * (2048 - 128 - len(label))  # Padding labels
                
                label = torch.tensor(np.array(label)).unsqueeze(0).to('cuda')
                input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0).to('cuda')
                 
                with torch.no_grad():
                    output_sentence = model.generate(input_sentence = query_text, input_image = image, input_mask = seg,  input_ids=input_ids,labels=label)
                    
                    print('input_sentence:',query_text)
                    print('output_sentence:',output_sentence)
                    print('grounded_sentence:',input_text[len(query_text):])
                    gt_sentence = input_text[len(query_text):]
                    
                    # Write the current row to the CSV file
                    writer.writerow([volume, image_path, output_sentence, gt_sentence])
                    file.flush()