
'''
Author: xm_cmic
Date: 2024-04-18 21:15:50
LastEditors: xm_cmic
LastEditTime: 2024-05-08 22:20:52
FilePath: /src-0508/model/global_model.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

from typing import Optional, Tuple

import numpy as np 
import pandas as pd

import os 
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from peft import LoraConfig, get_peft_model
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from model.modules import Prompt_Encoder

import transformers
from transformers import GPT2Config,GPT2Tokenizer,GPT2LMHeadModel
from transformers import BertPreTrainedModel, BertModel, BertTokenizer,AutoModel

class Global_VQA_Model(nn.Module):
    def __init__(self,tokenizer,language_model,vision_backbone='UNET',vision_pretrained=None,vision_learnable=True,text_dim = 4096):
        super().__init__()        
        self.vision_backbone = {
            'UNET' : PlainConvUNet(input_channels=1, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   deep_supervision=True,
                                   nonlin_first=False
                                   ),
        }[vision_backbone]
        if vision_pretrained:
            checkpoint = torch.load(vision_pretrained, map_location='cpu')
            new_state_dict = {k.replace('module.backbone.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            # Modify the problematic layers by averaging across the channel dimension
            for key in ['encoder.stages.0.0.convs.0.conv.weight', 'encoder.stages.0.0.convs.0.all_modules.0.weight']:
                weight = new_state_dict[key]
                averaged_weight = weight.mean(dim=1, keepdim=True)  # Averaging across the channel dimension
                new_state_dict[key] = averaged_weight
            self.vision_backbone.load_state_dict(new_state_dict,strict=False)
                
            model_keys = set(self.vision_backbone.state_dict().keys())
            loaded_keys = set(new_state_dict.keys())
            # 交集: 被加载的键
            loaded = model_keys & loaded_keys
            print("Loaded keys:", loaded)
            if vision_learnable:
                pass 
            else:
                for param in self.vision_backbone.parameters():
                    param.requires_grad = False
        else:
            self.vision_backbone.apply(InitWeights_He(1e-2))
        
        
        self.language_model = language_model
        
        self.tokenizer = tokenizer

        vis_dim = { # dim of latent embedding
            'UNET' : 768,
        }[vision_backbone]
        

        self.image_proj_mlp = {
            'UNET' : nn.Sequential(
                        nn.Linear(vis_dim, 1024),
                        nn.GELU(),
                        nn.Linear(1024, text_dim),
                        nn.GELU(),
                    ),
        }[vision_backbone]
        
        self.image_pool_layer = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
    
    def get_image_features(self, input_image):
        # Image Encoder and Pixel Decoder
        latent_embeddings, per_pixel_embeddings = self.vision_backbone(input_image)
        # pooled_embedding = self.image_pool_layer(latent_embeddings[-1])
        image_embedding = rearrange(latent_embeddings[-1], 'b dim h w d -> b (h w d) dim')
        image_tokens = self.image_proj_mlp(image_embedding)
        return image_tokens

    def forward(self,input_image,input_ids,labels= None):
        image_features = self.get_image_features(input_image).bfloat16()
        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        
        input_embedding = torch.cat([image_features,input_embedding], dim=1) 
        output = self.language_model(inputs_embeds = input_embedding, labels = labels)
        return output

    def generate(self,input_sentence,input_image,input_ids,labels):
        image_features = self.get_image_features(input_image).bfloat16()
        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([image_features,input_embedding], dim=1)
        output = self.language_model(inputs_embeds = input_embedding, labels = labels)
        print('Loss:',output.loss)
        
        
        model_inputs = self.tokenizer(input_sentence, return_tensors='pt').to(self.language_model.device)
        generate_input_ids = model_inputs['input_ids']
        generate_input_embedding = self.language_model.get_input_embeddings()(generate_input_ids)
        generate_input_embedding = torch.cat([image_features,generate_input_embedding], dim=1)
        
        with torch.no_grad():
            beam_output = self.language_model.generate(
                inputs_embeds = generate_input_embedding,
                max_new_tokens=512,
                num_beams=5,
                do_sample=False,
                early_stopping=True
            )
        output_sentence = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        return output_sentence
    
    def generate(self,input_sentence,input_image,input_ids,labels):
        image_features = self.get_image_features(input_image).bfloat16()
        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([image_features,input_embedding], dim=1)
        output = self.language_model(inputs_embeds = input_embedding, labels = labels)
        print('Loss:',output.loss)
        
        
        model_inputs = self.tokenizer(input_sentence, return_tensors='pt').to(self.language_model.device)
        generate_input_ids = model_inputs['input_ids']
        generate_input_embedding = self.language_model.get_input_embeddings()(generate_input_ids)
        generate_input_embedding = torch.cat([image_features,generate_input_embedding], dim=1)
        
        with torch.no_grad():
            beam_output = self.language_model.generate(
                inputs_embeds = generate_input_embedding,
                max_new_tokens=512,
                # num_beams=5,
                # do_sample=False,
                # early_stopping=True
            )
        output_sentence = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        return output_sentence
    
if __name__ == '__main__':
    model =  PlainConvUNet(input_channels=1, 
                                   n_stages=6, 
                                   features_per_stage=(64, 64, 128, 256, 512, 768), 
                                   conv_op=nn.Conv3d, 
                                   kernel_sizes=3, 
                                   strides=(1, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                                   n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                                   conv_bias=True, 
                                   norm_op=nn.InstanceNorm3d,
                                   norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                   dropout_op=None,
                                   dropout_op_kwargs=None,
                                   nonlin=nn.LeakyReLU, 
                                   nonlin_kwargs=None,
                                   deep_supervision=True,
                                   nonlin_first=False
                                   ).cuda()
    input_image = torch.rand((1, 1, 256, 256, 64)).cuda()
    latent_embeddings, per_pixel_embeddings = model(input_image)
    image_embedding = rearrange(latent_embeddings[-1], 'b dim h w d -> b (h w d) dim')
    print(image_embedding.shape)