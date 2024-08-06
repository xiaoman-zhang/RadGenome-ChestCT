'''
Author: xm_cmic
Date: 2024-06-06 13:52:33
LastEditors: xm_cmic
LastEditTime: 2024-06-16 19:10:23
FilePath: /src-0515/dataset/dataset_grounded_attn.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''


import os 
import csv 
import json 
import numpy as np 
import pandas as pd
import random
import copy
import nibabel as nib

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from monai.transforms import (
    Resized,
    Compose
)


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

def assign_sample_weights_df(df, column_name):
    freq_dict = df[column_name].value_counts().to_dict()
    max_freq = max(freq_dict.values())
    weights = {value: max_freq / freq for value, freq in freq_dict.items()}
    df['sample_weight'] = df[column_name].map(weights)
    return df


class CTRATE_Dataset(Dataset):
    def __init__(self,json_file,mask_root_dir,anatomy_mask_root_dir,tokenizer,crop_size=(256,256,64),train=True):
        # mask_root_dir big region
        self.mask_root_dir = mask_root_dir
        self.anatomy_mask_root_dir = anatomy_mask_root_dir
        self.task_list = ["report","abnormality","presence","location","size","disorder"]
        
        with open(json_file, 'r') as file:
            self.ctrate_data = json.load(file)
        self.volume_name_list = list(self.ctrate_data.keys())
        
        abnormality_template_json = './data/ctrate/generate_data/qa_templates/abnormality_template.json'
        with open(abnormality_template_json, 'r') as file:
            self.abnormality_template = json.load(file)['abnormality']
        
        location_template_json = './data/ctrate/generate_data/qa_templates/location_template.json'
        with open(location_template_json, 'r') as file:
            self.location_template = json.load(file)['location']
        
        presence_template_json = './data/ctrate/generate_data/qa_templates/presence_template.json'
        with open(presence_template_json, 'r') as file:
            self.presence_template = json.load(file)['presence']
        
        size_template_json = './data/ctrate/generate_data/qa_templates/size_template.json'
        with open(size_template_json, 'r') as file:
            self.size_template = json.load(file)['size']
            
        disorder_template_json = './data/ctrate/generate_data/qa_templates/disorders_template.json'
        with open(disorder_template_json, 'r') as file:
            self.disorders_template = json.load(file)['disorders']
        
        report_weight_dict_json = './RadGenome-ChestCT/radgenome_files/subset_2000/train_report_sentences_weights.json'
        with open(report_weight_dict_json, 'r') as file:
            self.report_weight_dict = json.load(file) 
        
        abnormality_weight_dict_json = './RadGenome-ChestCT/radgenome_files/subset_2000/train_abnormalities_sentences_weights.json'
        with open(abnormality_weight_dict_json, 'r') as file:
            self.abnormality_weight_dict = json.load(file) 
        
        location_weight_dict_json = './RadGenome-ChestCT/radgenome_files/subset_2000/train_location_sentences_weights.json'
        with open(location_weight_dict_json, 'r') as file:
            self.location_weight_dict = json.load(file) 
        
        self.crop_size = crop_size
        self.train = train
        self.img_padding = [-100]
        self.img_token_num = int((crop_size[0]//32)*(crop_size[1]//32)*(crop_size[2]//32))
        self.seq_length = 2048-(self.img_token_num)
        
        self.augmentator = Resized(
                spatial_size = [self.crop_size[0],self.crop_size[1],self.crop_size[2]],
                keys=["image","seg"],
                mode=['area',"area"]
            )

        
        self.tokenizer = tokenizer
    
    def __len__(self):
        if self.train:
            return len(self.volume_name_list)
        else:
            return 500
    
    def read_nii_img(self,nii_path):
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

    def select_reports(self,report_dict_list,report_weight_dict, select_number):
        if random.random() < 0.8:
            try:
                weights = [report_weight_dict.get(report["answer"])['cluster_weights'] for report in report_dict_list]
                # print(weights)
                # Select reports based on weights
                selected_items = random.choices(report_dict_list, weights=weights, k=select_number)
                # selected_items = random.sample(filtered_list, select_number)
            except ValueError:
                print('report number:',len(report_dict_list),',select_number:',select_number)
                # If there are not enough items to sample, use random.choices to allow repetition
                selected_items = random.choices(report_dict_list, k=select_number)
        else:
            selected_items = random.choices(report_dict_list, k=select_number)
        return selected_items
    
    def get_report_data(self,report_dict_list,abnormality_dict_list,volume,img_array,select_number):
        return_img_array_list = []
        return_mask_array_list = []
        return_query_text_list = []
        return_input_text_list = [] 
        
        non_select_regions = self.get_nonselect_regions(abnormality_dict_list)
        selected_items = self.select_reports(report_dict_list,self.report_weight_dict, select_number)
        
        for selected_item in selected_items:
            anatomy = selected_item['region']
            answer = selected_item['answer']
            node_anatomy = anatomy.split('/')[-1]
            head_anatomy = anatomy.split('/')[0]
            mask_array = np.ones(img_array.shape)
            if node_anatomy == "others" or node_anatomy == 'whole scan' or head_anatomy == "others" or head_anatomy == 'whole scan':
                pass 
            else:
                try:
                    mask_dir = os.path.join(self.anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                    mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                    if not os.path.exists(mask_path):
                        region_mask_dir = os.path.join(self.mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        head_anatomy = anatomy.split('/')[0]
                        mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                    mask_array = self.read_nii_img(mask_path)
                except:
                    pass 
            if node_anatomy == "others" or node_anatomy == 'whole scan' or head_anatomy == "others" or head_anatomy == 'whole scan':
                query_text = f'<Grounded report generation> Please describe the detailed finding of the {node_anatomy} region.'
                input_text = f'<Grounded report generation> Please describe the detailed finding of the {node_anatomy} region. {answer}'
            elif random.random() < 0.5:
                query_text = f'<Grounded report generation> Please describe the detailed finding of the {node_anatomy} region.'
                input_text = f'<Grounded report generation> Please describe the detailed finding of the {node_anatomy} region. {answer}'
            else:
                query_text = f'<Grounded report generation> Please describe the detailed finding of the given region.'
                input_text = f'<Grounded report generation> Please describe the detailed finding of the given region. <Region> The given region is {node_anatomy}. This is belong to {head_anatomy}. {answer}'
            
            return_img_array_list.append(img_array)
            return_query_text_list.append(query_text)
            return_input_text_list.append(input_text)
            return_mask_array_list.append(mask_array)
        return return_img_array_list,return_mask_array_list,return_query_text_list,return_input_text_list

    def get_nonselect_regions(self,abnormality_dict_list):
        filtered_list = [item for item in abnormality_dict_list if item["answer"] != "<Abnormality> No findings."]
        region_list = [item['region'] for item in filtered_list] 
        return region_list
    
    def select_abnormalities(self,abnormality_dict_list,abnormality_weight_dict, select_number):
        if random.random() < 0.8:
            try:
                weights = [abnormality_weight_dict.get(report["answer"])['cluster_weights'] for report in abnormality_dict_list]
                # print(weights)
                # Select reports based on weights
                selected_items = random.choices(abnormality_dict_list, weights=weights, k=select_number)
                # selected_items = random.sample(filtered_list, select_number)
            except ValueError:
                print('report number:',len(abnormality_dict_list),',select_number:',select_number)
                # If there are not enough items to sample, use random.choices to allow repetition
                selected_items = random.choices(abnormality_dict_list, k=select_number)
        else:
            selected_items = random.choices(abnormality_dict_list, k=select_number)
        return selected_items
    
    def select_location(self,location_dict_list,location_weight_dict, select_number):
        if random.random() < 0.8:
            try:
                weights = [location_weight_dict.get(report["answer"])['cluster_weights'] for report in location_dict_list]
                # print(weights)
                # Select reports based on weights
                selected_items = random.choices(location_dict_list, weights=weights, k=select_number)
                # selected_items = random.sample(filtered_list, select_number)
            except ValueError:
                print('report number:',len(location_dict_list),',select_number:',select_number)
                # If there are not enough items to sample, use random.choices to allow repetition
                selected_items = random.choices(location_dict_list, k=select_number)
        else:
            selected_items = random.choices(location_dict_list, k=select_number)
        return selected_items
    
    def get_abnormality_data(self,abnormality_dict_list,volume,img_array,select_number):
        return_img_array_list = []
        return_mask_array_list = []
        return_query_text_list = []
        return_input_text_list = [] 
        selected_items = self.select_abnormalities(abnormality_dict_list,self.abnormality_weight_dict, select_number)
        for selected_item in selected_items:
            anatomy = selected_item['region']
            answer = selected_item['answer']
            node_anatomy = anatomy.split('/')[-1]
            head_anatomy = anatomy.split('/')[0]
            mask_array = np.ones(img_array.shape)
            if node_anatomy == "others" or node_anatomy == 'whole scan' or head_anatomy == "others" or head_anatomy == 'whole scan':
                pass
            else:
                try:
                    mask_dir = os.path.join(self.anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                    mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                    if not os.path.exists(mask_path):
                        region_mask_dir = os.path.join(self.mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        head_anatomy = anatomy.split('/')[0]
                        mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                    mask_array = self.read_nii_img(mask_path)
                except:
                    pass 
            
            select_abnormality_template = random.choice(self.abnormality_template)
            if node_anatomy == "others":
                input_question = select_abnormality_template.replace('{region}',node_anatomy)
                query_text = f'<Grounded abnormality detection> {input_question}'
                input_text = f'<Grounded abnormality detection> {input_question} {answer}'
            if random.random() < 0.5:
                input_question = select_abnormality_template.replace('{region}',node_anatomy)
                query_text = f'<Grounded abnormality detection> {input_question}'
                input_text = f'<Grounded abnormality detection> {input_question} {answer}'
            else:
                input_question = select_abnormality_template.replace('{region}','given region')
                query_text = f'<Grounded abnormality detection> {input_question}'
                input_text = f'<Grounded abnormality detection> {input_question} <Region> The given region is {node_anatomy}. This is belong to {head_anatomy}. {answer}'
 
            return_img_array_list.append(img_array)
            return_query_text_list.append(query_text)
            return_input_text_list.append(input_text)
            return_mask_array_list.append(mask_array)
        return return_img_array_list,return_mask_array_list,return_query_text_list,return_input_text_list

    
    def get_presence_data(self,presence_dict_list,volume,img_array,select_number):
        return_img_array_list = []
        return_mask_array_list = []
        return_query_text_list = []
        return_input_text_list = [] 
        try:
            selected_items = random.sample(presence_dict_list, select_number)
        except:
            selected_items = random.choices(presence_dict_list, k=select_number)
         
        for selected_item in selected_items:
            anatomy = selected_item['region']
            answer = selected_item['answer']
            abnormality = selected_item['abnormality']
            node_anatomy = anatomy.split('/')[-1]
            head_anatomy = anatomy.split('/')[0]
            mask_array = np.ones(img_array.shape)
            if node_anatomy == "others" or node_anatomy == 'whole scan' or head_anatomy == "others" or head_anatomy == 'whole scan':
                pass 
            else:
                try:
                    mask_dir = os.path.join(self.anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                    mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                    if not os.path.exists(mask_path):
                        region_mask_dir = os.path.join(self.mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        head_anatomy = anatomy.split('/')[0]
                        mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                    mask_array = self.read_nii_img(mask_path)
                except:
                    pass
            
            select_presence_template = random.choice(self.presence_template)
            if random.random() < 0.5:
                input_question = select_presence_template.replace('{region}',node_anatomy).replace('{abnormality}',abnormality)
            else:
                input_question = select_presence_template.replace('{region}','given region').replace('{abnormality}',abnormality)
            
            query_text = f'<Grounded abnormality presence> {input_question}'
            input_text = f'<Grounded abnormality presence> {input_question} {answer}'
        
            return_img_array_list.append(img_array)
            return_query_text_list.append(query_text)
            return_input_text_list.append(input_text)
            return_mask_array_list.append(mask_array)
        return return_img_array_list,return_mask_array_list,return_query_text_list,return_input_text_list

    def get_location_data(self,location_dict_list,volume,img_array,select_number):
        return_img_array_list = []
        return_mask_array_list = []
        return_query_text_list = []
        return_input_text_list = [] 
        selected_items = self.select_abnormalities(location_dict_list,self.location_weight_dict, select_number)
        # selected_items = random.sample(location_dict_list, select_number)
        # try:
        #     selected_items = random.sample(location_dict_list, select_number)
        # except:
        #     selected_items = random.choices(location_dict_list, k=select_number)
         
        for selected_item in selected_items:
            anatomy = selected_item['region']
            answer = selected_item['answer']
            abnormality = selected_item['abnormality']
            node_anatomy = anatomy.split('/')[-1]
            head_anatomy = anatomy.split('/')[0]
            print(node_anatomy,head_anatomy)
            # new_answer = 'The head anatomy is ' + head_anatomy + '. The node anatomy is ' + node_anatomy + '.'
            mask_array = np.ones(img_array.shape)
            if node_anatomy == "others" or node_anatomy == 'whole scan' or head_anatomy == "others" or head_anatomy == 'whole scan':
                pass 
            else:
                try:
                    mask_dir = os.path.join(self.anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                    mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                    if not os.path.exists(mask_path):
                        region_mask_dir = os.path.join(self.mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        head_anatomy = anatomy.split('/')[0]
                        mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                    mask_array = self.read_nii_img(mask_path)
                except:
                    pass
            
            select_location_template = random.choice(self.location_template)
            input_question = select_location_template.replace('{abnormality}',abnormality)
            query_text = f'<Abnormality location> {input_question}'
            input_text = f'<Abnormality location> {input_question} {answer}'
          
            return_img_array_list.append(img_array)
            return_query_text_list.append(query_text)
            return_input_text_list.append(input_text)
            return_mask_array_list.append(mask_array)
        return return_img_array_list,return_mask_array_list,return_query_text_list,return_input_text_list

    def get_size_data(self,size_dict_list,volume,img_array,select_number):
        return_img_array_list = []
        return_mask_array_list = []
        return_query_text_list = []
        return_input_text_list = [] 
        try:
            selected_items = random.sample(size_dict_list, select_number)
        except:
            selected_items = random.choices(size_dict_list, k=select_number)
        
        # selected_items = random.sample(size_dict_list, select_number)
        for selected_item in selected_items:
            anatomy = selected_item['region']
            answer = selected_item['answer']
            abnormality = selected_item['abnormality']
            node_anatomy = anatomy.split('/')[-1]
            head_anatomy = anatomy.split('/')[0]
            mask_array = np.ones(img_array.shape)
            if node_anatomy == "others" or node_anatomy == 'whole scan' or head_anatomy == "others" or head_anatomy == 'whole scan':
                pass
            else:
                try:
                    mask_dir = os.path.join(self.anatomy_mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                    mask_path = os.path.join(mask_dir,node_anatomy+'.nii.gz')
                    if not os.path.exists(mask_path):
                        region_mask_dir = os.path.join(self.mask_root_dir,'seg_'+volume.replace('.nii.gz',''))
                        head_anatomy = anatomy.split('/')[0]
                        mask_path = os.path.join(region_mask_dir,head_anatomy+'.nii.gz')
                    mask_array = self.read_nii_img(mask_path)
                except:
                    pass
                    
            select_size_template = random.choice(self.size_template)
            if random.random() < 0.5:
                input_question = select_size_template.replace('{abnormality}',abnormality).replace('{region}',node_anatomy)
            else:
                input_question = select_size_template.replace('{abnormality}',abnormality).replace('{region}','given region')
                
            query_text = f'<Abnormality size prediction> {input_question}'
            input_text = f'<Abnormality size prediction> {input_question} {answer}'
         
            return_img_array_list.append(img_array)
            return_query_text_list.append(query_text)
            return_input_text_list.append(input_text)
            return_mask_array_list.append(mask_array)
        return return_img_array_list,return_mask_array_list,return_query_text_list,return_input_text_list

    def get_text_query(self,input_text,query_text):
        # Tokenize input text without adding special tokens
        tokenizer_output = self.tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
        input_ids = tokenizer_output['input_ids']
    
        # Copy input_ids to label and set eos_token_id locations to -100
        label = input_ids.clone()

        # Process query text if provided
        if query_text:
            query_tokenizer_output = self.tokenizer(query_text, add_special_tokens=False, return_tensors="pt")
            query_input_ids = query_tokenizer_output['input_ids']
            # Set the labels for the length of the query input ids to -100
            label[:, :query_input_ids.size(1)] = -100
        
        # Convert input_ids to list and append eos_token_id, then pad to 960 (2048 - 64) as per your requirement
        input_ids = input_ids.squeeze().tolist() + [self.tokenizer.eos_token_id]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (2048- self.img_token_num - len(input_ids))  # Padding input_ids

        # Convert label to list, append eos_token_id, and pad both to 2048
        label = label.squeeze().tolist() + [self.tokenizer.eos_token_id]
        label = [-100]*self.img_token_num + label + [-100] * (2048- self.img_token_num - len(label))  # Padding labels
        return np.array(input_ids),np.array(label)

    def process_texts(self, input_texts, query_texts):
        # Tokenize input texts without adding special tokens
        tokenizer_output = self.tokenizer(input_texts, add_special_tokens=False, return_tensors="pt", padding=True)
        input_ids = tokenizer_output['input_ids']

        # Initialize labels as a copy of input_ids
        labels = input_ids.clone()

        # Tokenize query texts without adding special tokens
        for i in range(len(query_texts)):
            query_text = query_texts[i]
            query_tokenizer_output = self.tokenizer(query_texts, add_special_tokens=False, return_tensors="pt", padding=True)
            query_input_ids = query_tokenizer_output['input_ids']
            # Set the labels for the length of the query input ids to -100
            labels[i, :query_input_ids.size(1)] = -100
        
        # query_tokenizer_output = self.tokenizer(query_texts, add_special_tokens=False, return_tensors="pt", padding=True)
        # query_input_ids = query_tokenizer_output['input_ids']
        
        # # Set the labels for the length of the query input ids to -100
        # for i in range(query_input_ids.size(0)):
        #     labels[i, :query_input_ids.size(1)] = -100
    
        
        # Convert input_ids to list, append eos_token_id, and pad
        input_ids = input_ids.tolist()
        for i in range(len(input_ids)):
            input_ids[i].append(self.tokenizer.eos_token_id)
            input_ids[i].extend([self.tokenizer.pad_token_id] * (2048 - self.img_token_num - len(input_ids[i])))
        
        # Convert labels to list, append eos_token_id, and pad
        labels = labels.tolist()
        for i in range(len(labels)):
            labels[i].append(self.tokenizer.eos_token_id)
            labels[i] = [-100] * self.img_token_num + labels[i] + [-100] * (2048 - self.img_token_num - len(labels[i]))
        
        return input_ids, labels
    
    def __getitem__(self, idx):
        # 每次选取一个image 
        volume = self.volume_name_list[idx]
        volume_data = self.ctrate_data[volume]
        image_path = volume_data['image_path']
        img_array = self.read_nii_img(image_path)
        img_array = zscore_process(img_array)
        # img_array = img_array[np.newaxis,:,:]
        
        image_array_list = []
        mask_array_list = []
        query_text_list = []
        input_text_list = [] 
        
        volume_tasks = volume_data['task']
        # ["report","abnormality","presence","location","size","disorder"]
        # for each volume, select 3 report, 2 abnormality, 1 presence, 1 location, 1 size
        num_report = 2
        num_abnormality = 1
        num_presence = 1
        num_location = 3
        num_size = 1
        if len(volume_tasks['location']) == 0:
            num_report = 5
            num_location = 0
        if  len(volume_tasks['size']) == 0:
            num_abnormality = 2
            num_size = 0
        
        report_img_array_list,report_mask_array_list,report_query_text_list,report_input_text_list = self.get_report_data(volume_tasks['report'],volume_tasks['abnormalities'],volume,img_array,num_report)
        abnormality_img_array_list,abnormality_mask_array_list,abnormality_query_text_list,abnormality_input_text_list = self.get_abnormality_data(volume_tasks['abnormalities'],volume,img_array,num_abnormality)
        presence_img_array_list,presence_mask_array_list,presence_query_text_list,presence_input_text_list = self.get_presence_data(volume_tasks['presence'],volume,img_array,num_presence)
        image_array_list.extend(report_img_array_list)
        image_array_list.extend(abnormality_img_array_list)
        image_array_list.extend(presence_img_array_list)
        mask_array_list.extend(report_mask_array_list)
        mask_array_list.extend(abnormality_mask_array_list)
        mask_array_list.extend(presence_mask_array_list)
        query_text_list.extend(report_query_text_list)
        query_text_list.extend(abnormality_query_text_list)
        query_text_list.extend(presence_query_text_list)
        input_text_list.extend(report_input_text_list)
        input_text_list.extend(abnormality_input_text_list)
        input_text_list.extend(presence_input_text_list)
        
        if num_location != 0:
            location_img_array_list,location_mask_array_list,location_query_text_list,location_input_text_list = self.get_location_data(volume_tasks['location'],volume,img_array,num_location)
            image_array_list.extend(location_img_array_list)
            mask_array_list.extend(location_mask_array_list)
            query_text_list.extend(location_query_text_list)
            input_text_list.extend(location_input_text_list)
        if num_size != 0:
            size_img_array_list,size_mask_array_list,size_query_text_list,size_input_text_list = self.get_size_data(volume_tasks['size'],volume,img_array,num_size)
            image_array_list.extend(size_img_array_list)
            mask_array_list.extend(size_mask_array_list)
            query_text_list.extend(size_query_text_list)
            input_text_list.extend(size_input_text_list)
        
        img_array = np.array(image_array_list)
        try:
            mask_array = np.array(mask_array_list)
        except:
            mask_array = np.ones(img_array.shape)
            for i, element in enumerate(mask_array_list):
                print(f"Element {i} shape: {np.shape(element)}")
                if element.shape == img_array[i].shape:
                    mask_array[i] = element

        # img_array = np.array(image_array_list)
        # mask_array = np.array(mask_array_list)
        data_dict = {'image': img_array, 'seg':mask_array}
        aug_data_dict = self.augmentator(data_dict)
        image = aug_data_dict['image']
        seg = aug_data_dict['seg']
        input_ids, labels = self.process_texts(input_text_list,query_text_list)
        item = {
                "input_image": image,
                "input_mask": seg,
                'input_ids': np.array(input_ids),    
                'labels': np.array(labels)
            }
        return item 
        
