'''
Author: xm_cmic
Date: 2024-06-12 15:25:33
LastEditors: xm_cmic
LastEditTime: 2024-06-13 13:54:01
FilePath: /src-0515/radgenome_files/subset_2000/assign_weights.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import torch
import json 
from transformers import AutoTokenizer, AutoModel

def get_medcpt_embedding(tokenizer,model,sentence):
    with torch.no_grad():
        encoded = tokenizer(
            [sentence], 
            truncation=True, 
            padding=True, 
            return_tensors='pt', 
            max_length=64,
        )
        
        # encode the queries (use the [CLS] last hidden states as the representations)
        embeds = model(**encoded).last_hidden_state[:, 0, :].detach()
    return embeds

def cluster_sentences(sentence_dict, num_clusters=1000):
    language_model = '/mnt/petrelfs/zhangxiaoman/CODE/2024_RadEval/models/MedCPT-Query-Encoder'
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    model = AutoModel.from_pretrained(language_model)
    
    print('Load Embeddings')
    # 提取句子和对应的计数
    sentences = list(sentence_dict.keys())
    counts = list(sentence_dict.values())
    
    # 编码句子
    embeddings = np.array(torch.cat([get_medcpt_embedding(tokenizer,model,sentence) for sentence in sentences]))
    # 聚类
    print('Clustering')
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 计算每个簇的权重
    cluster_weights = defaultdict(float)
    for cluster_label, count in zip(cluster_labels, counts):
        cluster_weights[cluster_label] += count
    
    # 对权重进行归一化
    total_count = sum(counts)
   
    return cluster_labels, cluster_weights,total_count

def read_json_and_get_sentences(json_file,target):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 初始化句子字典
    sentence_dict = defaultdict(int)
    
    # 遍历每个样本
    for sample, sample_data in data.items():
        # 获取样本的报告列表
        reports = sample_data['task'][target]
        # 遍历每个报告
        for report in reports:
            # 提取报告中的句子
            sentence = report['answer']
            # 将每个句子添加到句子字典中并计数
            if sentence:
                sentence_dict[sentence] += 1
    sentence_dict = dict(sorted(sentence_dict.items(), key=lambda x: x[1], reverse=True))
    return sentence_dict

# # 示例句子字典
# target='report'
# sentence_dict = read_json_and_get_sentences('train.json',target)
# # 将句子字典保存到 JSON 文件
# with open(f'train_{target}_sentences.json', 'w') as f:
#     json.dump(sentence_dict, f, indent=4)


# # 聚类并获取权重
# cluster_labels, cluster_weights, total_count = cluster_sentences(sentence_dict,num_clusters=1000)
# cluster_labels = cluster_labels.tolist()

# save_cluster_weights = {}
# new_cluster_weights = cluster_weights   
# for cluster_label in cluster_weights:
#     new_weights = total_count / cluster_weights[cluster_label]
#     print(cluster_weights[cluster_label],total_count,new_weights)
#     if new_weights > 10000:
#         new_weights = 10000
#     new_cluster_weights[cluster_label] = new_weights
#     save_cluster_weights[str(cluster_label)] = cluster_weights[cluster_label]
    
# # 将聚类标签和权重添加到同一个 JSON 文件中
# save_sentence_dict = {}
# for idx, sentence in enumerate(sentence_dict):
#     save_sentence_dict[sentence] = {
#         'count': sentence_dict[sentence],
#         'cluster_labels': cluster_labels[idx],
#         'cluster_weights': new_cluster_weights[cluster_labels[idx]]
# }
    
# with open(f'train_{target}_sentences_weights.json', 'w') as f:
#     json.dump(save_sentence_dict, f, indent=4)
    
# # 将聚类标签和权重添加到同一个 JSON 文件中
# with open(f'train_{target}_sentences.json', 'r+') as f:
#     data = json.load(f)
#     data['cluster_labels'] = cluster_labels
#     data['cluster_weights'] = dict(save_cluster_weights)
#     f.seek(0)
#     json.dump(data, f, indent=4)
 
 
 
 # 示例句子字典
target='location'
sentence_dict = read_json_and_get_sentences('train.json',target)
# 将句子字典保存到 JSON 文件
with open(f'train_{target}_sentences.json', 'w') as f:
    json.dump(sentence_dict, f, indent=4)


# 聚类并获取权重
cluster_labels, cluster_weights, total_count = cluster_sentences(sentence_dict,num_clusters=100)
cluster_labels = cluster_labels.tolist()

new_cluster_weights = cluster_weights   
save_cluster_weights = {}
for cluster_label in cluster_weights:
    new_weights = total_count / cluster_weights[cluster_label]
    if new_weights > 10000:
        new_weights = 10000
    new_cluster_weights[cluster_label] = new_weights
    save_cluster_weights[str(cluster_label)] = cluster_weights[cluster_label]
        
# 将聚类标签和权重添加到同一个 JSON 文件中
save_sentence_dict = {}
for idx, sentence in enumerate(sentence_dict):
    save_sentence_dict[sentence] = {
        'count': sentence_dict[sentence],
        'cluster_labels': cluster_labels[idx],
        'cluster_weights': new_cluster_weights[cluster_labels[idx]]
}
    
with open(f'train_{target}_sentences_weights.json', 'w') as f:
    json.dump(save_sentence_dict, f, indent=4)
    
# 将聚类标签和权重添加到同一个 JSON 文件中
with open(f'train_{target}_sentences.json', 'r+') as f:
    data = json.load(f)
    data['cluster_labels'] = cluster_labels
    data['cluster_weights'] = dict(save_cluster_weights)
    f.seek(0)
    json.dump(data, f, indent=4)
 
#  srun -p medai -x SH-IDC1-10-140-0-168   --gpus-per-task=1  --cpus-per-task 24 python assign_weights.py 