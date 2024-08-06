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
from transformers import GPT2Config,GPT2Tokenizer,GPT2LMHeadModel
from transformers import BertPreTrainedModel, BertModel, BertTokenizer,AutoModel

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
    
    
    def get_image_features(self, input_image):
        # Image Encoder and Pixel Decoder
        latent_embeddings, per_pixel_embeddings = self.vision_backbone(input_image)
        # [torch.Size([1, 64, 256, 256, 64]), torch.Size([1, 64, 128, 128, 32]), torch.Size([1, 128, 64, 64, 16]), torch.Size([1, 256, 32, 32, 8]), torch.Size([1, 512, 16, 16, 4])]
        # pooled_embedding = self.image_pool_layer(latent_embeddings[-1])
        image_embedding = rearrange(latent_embeddings[-1], 'b dim h w d -> b (h w d) dim')
        # image_tokens = self.image_proj_mlp(image_embedding)
        return image_embedding

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

        attn_mask_features,_ = self.multihead_attn(query=mask_features, key=image_features, value=image_features)
        image_tokens = self.image_proj_mlp(attn_mask_features)
        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        # image_tokens torch.Size([8, 128, 4096])
        # input_embedding torch.Size([8, 1, 1920, 4096])
        input_embedding = torch.cat([image_tokens,input_embedding], dim=1) 
        output = self.language_model(inputs_embeds = input_embedding, labels = labels)
        return output
    
    def generate(self,input_sentence,input_image,input_mask,input_ids,labels):
        image_features = self.get_image_features(input_image)#.bfloat16() # 1 l d
        mask_features = self.get_mask_features(input_mask)#.bfloat16() # b l d

        attn_mask_features,_ = self.multihead_attn(query=mask_features, key=image_features, value=image_features)
        image_tokens = self.image_proj_mlp(attn_mask_features).bfloat16()
        input_embedding = self.language_model.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([image_tokens,input_embedding], dim=1) 
       
        output = self.language_model(inputs_embeds = input_embedding, labels = labels)
        print('Loss:',output.loss)
        
        model_inputs = self.tokenizer(input_sentence, return_tensors='pt').to(self.language_model.device)
        generate_input_ids = model_inputs['input_ids']
        generate_input_embedding = self.language_model.get_input_embeddings()(generate_input_ids)
        generate_input_embedding = torch.cat([image_tokens,generate_input_embedding], dim=1)
        
        with torch.no_grad():
            beam_output = self.language_model.generate(
                inputs_embeds = generate_input_embedding,
                max_new_tokens=512,
                num_beams=5,
                do_sample=True,
                early_stopping=True
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
    # input_mask = torch.rand((1, 1, 256, 256, 64)).cuda()
    # input_ids = torch.zeros((1,900),dtype=torch.int32).to('cuda')
    # labels = torch.zeros((1,900+64),dtype=torch.long).to('cuda')
    latent_embeddings, per_pixel_embeddings = model(input_image)
    print([per_pixel_embedding.shape for per_pixel_embedding in per_pixel_embeddings])
    print([latent_embedding.shape for latent_embedding in latent_embeddings])
    image_embedding = rearrange(latent_embeddings[-1], 'b dim h w d -> b (h w d) dim')
    print([image_embedding.shape for image_embedding in latent_embeddings])