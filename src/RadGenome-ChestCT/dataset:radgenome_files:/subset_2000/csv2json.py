'''
Author: xm_cmic
Date: 2024-06-06 15:22:55
LastEditors: xm_cmic
LastEditTime: 2024-06-07 17:44:45
FilePath: /src-0515/radgenome_files/subset_2000/csv2json.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import os 
import csv 
import json 
import pandas as pd 
from tqdm import tqdm 

def get_report_volume_dict(qa_df_regioned_report,volume):
    return_report_volume_dict = []
    volume_report_df = qa_df_regioned_report[qa_df_regioned_report['Volumename'] == volume]
    # 逐行打印筛选出的行
    for index, row in volume_report_df.iterrows():
        is_nan = pd.isna(row['Anatomy'])
        if is_nan:
            region = 'whole scan'
        else:
            region = row['Anatomy']
        answer = ' <Report> ' + row['Sentence']
        report_volume_dict_index = {
            "region": region,
            "answer": answer
        }
        return_report_volume_dict.append(report_volume_dict_index)
    return return_report_volume_dict

def get_abnormalities_volume_dict(qa_df_with_abnormalities,volume):
    return_abnormalities_volume_dict = []
    volume_abnormalities_df = qa_df_with_abnormalities[qa_df_with_abnormalities['Volumename'] == volume]
    # 逐行打印筛选出的行
    for index, row in volume_abnormalities_df.iterrows():
        is_nan = pd.isna(row['Anatomy'])
        if is_nan:
            region = 'whole scan'
        else:
            region = row['Anatomy']
        abnormality = row['Abnormality']
        answer = f'<Abnormality> {abnormality.capitalize()}.'
        abnormalities_volume_dict_index = {
            "region": region,
            "answer": answer
        }
        return_abnormalities_volume_dict.append(abnormalities_volume_dict_index)
    return return_abnormalities_volume_dict

def get_location_volume_dict(qa_df_with_location,volume):
    return_location_volume_dict = []
    volume_location_df = qa_df_with_location[qa_df_with_location['Volumename'] == volume]
    # 逐行打印筛选出的行wo
    for index, row in volume_location_df.iterrows():
        is_nan = pd.isna(row['Anatomy'])
        if is_nan:
            region = 'whole scan'
        else:
            region = row['Anatomy']
        abnormality = row['Abnormality']
        answer = f'<Location> {region.capitalize()}.'
        location_volume_dict_index = {
            "region": region,
            "abnormality": abnormality,
            "answer": answer
        }
        return_location_volume_dict.append(location_volume_dict_index)
    return return_location_volume_dict

def get_presence_volume_dict(qa_df_with_presence,volume):
    return_presence_volume_dict = []
    volume_presence_df = qa_df_with_presence[qa_df_with_presence['Volumename'] == volume]
    # 逐行打印筛选出的行wo
    for index, row in volume_presence_df.iterrows():
        is_nan = pd.isna(row['Anatomy'])
        if is_nan:
            region = 'whole scan'
        else:
            region = row['Anatomy']
        presence = row['Presence']
        abnormality = row['Finding']
        answer = f'<Presence> {presence}'
        presence_volume_dict_index = {
            "region": region,
            "abnormality": abnormality,
            "answer": answer
        }
        return_presence_volume_dict.append(presence_volume_dict_index)
    return return_presence_volume_dict

def get_size_volume_dict(qa_df_with_size,volume):
    return_size_volume_dict = []
    volume_size_df = qa_df_with_size[qa_df_with_size['Volumename'] == volume]
    # 逐行打印筛选出的行wo
    for index, row in volume_size_df.iterrows():
        is_nan = pd.isna(row['Anatomy'])
        if is_nan:
            region = 'whole scan'
        else:
            region = row['Anatomy']
        size = row['Size']
        abnormality = row['Abnormality']
        answer = f'<Size> {size}.'
        size_volume_dict_index = {
            "region": region,
            "abnormality": abnormality,
            "answer": answer
        }
        return_size_volume_dict.append(size_volume_dict_index)
    return return_size_volume_dict

def assign_sample_weights_df(df, column_name):
    freq_dict = df[column_name].value_counts().to_dict()
    max_freq = max(freq_dict.values())
    weights = {value: max_freq / freq for value, freq in freq_dict.items()}
    df['sample_weight'] = df[column_name].map(weights)
    return df


def csv2json(save_json_file,image_path_csv,regioned_report_csv,qa_abnormality_csv,qa_location_csv,qa_presence_csv,qa_size_csv,disorders_csv,mask_root_dir,anatomy_mask_root_dir):
    save_data_dict = {}
    report_df = pd.read_csv(image_path_csv)
    volume_to_nii_path_dict = dict(zip(report_df['Volumename'], report_df['nii_path']))
    qa_df_regioned_report = pd.read_csv(regioned_report_csv)
    qa_df_with_abnormalities = pd.read_csv(qa_abnormality_csv)
    qa_df_with_abnormalities = assign_sample_weights_df(qa_df_with_abnormalities,'Abnormality')
    
    qa_df_with_location = pd.read_csv(qa_location_csv)
    qa_df_with_presence = pd.read_csv(qa_presence_csv)
    
    qa_size_df = pd.read_csv(qa_size_csv)
    qa_disorder_df = pd.read_csv(disorders_csv) 
    for volume in tqdm(volume_to_nii_path_dict):
        save_volume_dict = {}
        image_path = volume_to_nii_path_dict[volume]
        save_volume_dict['image_path'] = image_path
        save_task_dict = {}
        report_volume_dict = get_report_volume_dict(qa_df_regioned_report,volume)
        abnormalities_volume_dict = get_abnormalities_volume_dict(qa_df_with_abnormalities,volume)
        location_volume_dict = get_location_volume_dict(qa_df_with_location,volume)
        presence_volume_dict = get_presence_volume_dict(qa_df_with_presence,volume)
        size_volume_dict = get_size_volume_dict(qa_size_df,volume)
        save_task_dict = {
            "report": report_volume_dict,
            "abnormalities": abnormalities_volume_dict,
            "location": location_volume_dict,
            "presence": presence_volume_dict,
            "size": size_volume_dict
        }
        save_volume_dict['task'] = save_task_dict
        save_data_dict[volume] = save_volume_dict
    with open(save_json_file, 'w') as file:
        json.dump(save_data_dict, file, indent=4)
        

save_json_file = 'train.json'

mask_root_dir = '/mnt/hwfile/medai/zhangxiaoman/DATA/CT-RATE/SAT/train_region_mask'
anatomy_mask_root_dir ='/mnt/hwfile/medai/zhangxiaoman/DATA/CT-RATE/SAT/train_anatomy_mask'

csv2json(save_json_file,'train_image_path.csv','train_region_report.csv','train_vqa_abnormality.csv','train_vqa_location.csv','train_vqa_presence.csv','train_vqa_size.csv','train_case_disorders.csv',mask_root_dir,anatomy_mask_root_dir)
# image_path_csv,regioned_report_csv,qa_abnormality_csv,qa_location_csv,qa_presence_csv,qa_size_csv,disorders_csv,mask_root_dir,anatomy_mask_root_dir
