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
# from modules import Prompt_Encoder

import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

class Grounded_VQA_Model_Attn(nn.Module):
    def __init__(self,tokenizer,language_model,vision_backbone='UNET',vision_pretrained=None,vision_learnable = True,text_dim = 4096):
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
        

        self.prompt_encoder = Prompt_Encoder(vis_dim)
        self.hierarchy_mlp = {
            'UNET' : nn.Sequential(
                        nn.Linear(1024+vis_dim, 1024),
                        nn.GELU(),
                        nn.Linear(1024, vis_dim),
                        nn.GELU(),
                    ),
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
        self.multihead_attn = nn.MultiheadAttention(vis_dim, num_heads=4,batch_first=True)
    
    # Function to downsample embeddings to the target shape
    def downsample_embeddings(self,embeddings, target_shape=(16, 16, 4)):
        downsampled_embeddings = []
        for emb in embeddings:
            # Extract the batch size and number of channels
            batch_size, channels, depth, height, width = emb.shape
            # Downsample the spatial dimensions to target_shape
            emb_downsampled = F.interpolate(emb, size=target_shape, mode='trilinear', align_corners=False)
            downsampled_embeddings.append(emb_downsampled)
        return downsampled_embeddings


    def get_image_features(self, input_image):
        # Image Encoder and Pixel Decoder
        latent_embeddings, per_pixel_embeddings = self.vision_backbone(input_image)
        # [torch.Size([1, 64, 256, 256, 64]), torch.Size([1, 64, 128, 128, 32]), torch.Size([1, 128, 64, 64, 16]), torch.Size([1, 256, 32, 32, 8]), torch.Size([1, 512, 16, 16, 4])]
        # Downsample the embeddings
        downsampled_latent_embeddings = self.downsample_embeddings(latent_embeddings, target_shape=(8,8,2))
        # Concatenate along dim1 (channels)
        concatenated_embeddings = torch.cat(downsampled_latent_embeddings, dim=1)
        image_embedding = rearrange(concatenated_embeddings, 'b dim h w d -> b (h w d) dim')
        image_tokens = self.hierarchy_mlp(image_embedding)
        return image_tokens

    def get_mask_features(self, input_mask):
        # Prompt Encoder for mask
        mask_embeddings = self.prompt_encoder(input_mask)
        mask_tokens = rearrange(mask_embeddings, 'b dim h w d -> b (h w d) dim')
        return mask_tokens
    
    def forward(self,input_image,input_mask,input_ids,labels= None):
        # input_image: torch.Size([1, 1, 256, 256, 64])
        # torch.Size([1, bs, 256, 256, 64])
        # torch.Size([1, bs, 1920])
        # torch.Size([1, bs, 2048])
        input_image = rearrange(input_image, 'c b h w d -> b c h w d')
        input_mask = rearrange(input_mask, 'c b h w d -> b c h w d')
        input_ids = rearrange(input_ids, 'c b d -> (c b) d')
        labels = rearrange(labels, 'c b d -> (c b) d')
        
        image_features = self.get_image_features(input_image).bfloat16() # 1 l d
        mask_features = self.get_mask_features(input_mask).bfloat16() # b l d
        
        # (2048x1792 and 1280x1024)
        attn_mask_features,_ = self.multihead_attn(query=mask_features, key=image_features, value=image_features)
        image_tokens = self.image_proj_mlp(attn_mask_features)
        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        # image_tokens torch.Size([8, 128, 4096])
        # input_embedding torch.Size([8, 1, 1920, 4096])
        input_embedding = torch.cat([image_tokens,input_embedding], dim=1) 
        output = self.language_model(inputs_embeds = input_embedding, labels = labels)
        return output
    
    def generate(self,input_sentence,input_image,input_mask,input_ids,labels):
        image_features = self.get_image_features(input_image).bfloat16()
        mask_features = self.get_mask_features(input_mask).bfloat16()
        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([image_features + mask_features,input_embedding], dim=1)
        output = self.language_model(inputs_embeds = input_embedding, labels = labels)
        print('Loss:',output.loss)
        
        model_inputs = self.tokenizer(input_sentence, return_tensors='pt').to(self.language_model.device)
        generate_input_ids = model_inputs['input_ids']
        generate_input_embedding = self.language_model.get_input_embeddings()(generate_input_ids)
        generate_input_embedding = torch.cat([image_features + mask_features,generate_input_embedding], dim=1)
        
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

 
if __name__ == '__main__':
    # model =  PlainConvUNet(input_channels=1, 
    #                                n_stages=6, 
    #                                features_per_stage=(64, 64, 128, 256, 512, 768), 
    #                                conv_op=nn.Conv3d, 
    #                                kernel_sizes=3, 
    #                                strides=(1, 2, 2, 2, 2, 2), 
    #                                n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
    #                                n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
    #                                conv_bias=True, 
    #                                norm_op=nn.InstanceNorm3d,
    #                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
    #                                dropout_op=None,
    #                                dropout_op_kwargs=None,
    #                                nonlin=nn.LeakyReLU, 
    #                                nonlin_kwargs=None,
    #                                deep_supervision=True,
    #                                nonlin_first=False
    #                                ).cuda()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    language_model = AutoModelForCausalLM.from_pretrained('/mnt/hwfile/medai/LLMModels/Model/Meta-Llama-3-8B-Instruct', device_map="cpu",torch_dtype=torch.bfloat16)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    
    language_model = get_peft_model(language_model, config)
    language_model.print_trainable_parameters()
    
    language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    language_model.enable_input_require_grads()
    language_model.config.use_cache = False
    model = Grounded_VQA_Model_Attn(tokenizer,language_model,vision_backbone='UNET',vision_pretrained='/mnt/petrelfs/zhangxiaoman/CODE/2024_CTRG/data/ctrate/sat_preprocessing/seg_model_version1/checkpoint/SAT_Nano.pth',vision_learnable=False,text_dim=2048).cuda()
    
    input_image = torch.rand((1, 1, 256, 256, 64)).cuda()
    input_mask = torch.rand((1, 1, 256, 256, 64)).cuda()
    input_ids = torch.zeros((1,1,900),dtype=torch.int32).cuda()
    labels = torch.zeros((1,1,900+64),dtype=torch.long).cuda()
    output = model(input_image,input_mask,input_ids,labels)
    
    # latent_embeddings, per_pixel_embeddings = model(input_image)
    # print([per_pixel_embedding.shape for per_pixel_embedding in per_pixel_embeddings])
    # print([latent_embedding.shape for latent_embedding in latent_embeddings])
    # image_embedding = rearrange(latent_embeddings[-1], 'b dim h w d -> b (h w d) dim')
    # print([image_embedding.shape for image_embedding in latent_embeddings])