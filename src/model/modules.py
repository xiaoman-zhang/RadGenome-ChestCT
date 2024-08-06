'''
Author: xm_cmic
Date: 2024-04-29 15:46:59
LastEditors: xm_cmic
LastEditTime: 2024-05-02 10:46:00
FilePath: /src-0502/model/modules.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class Prompt_Encoder(nn.Module):
    def __init__(self,text_dim=768):
        super(Prompt_Encoder, self).__init__()
        # 假设输入通道数为1，我们需要适当调整卷积层的输出通道数以匹配最终输出维度
        self.downsample = nn.Upsample(scale_factor=0.25, mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(1, 128, kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv3d(128, 512, kernel_size=(2, 2, 2), stride=2)
        self.conv3 = nn.Conv3d(512, text_dim, kernel_size=(2, 2, 2), stride=2)
        # self.conv4 = nn.Conv3d(256, text_dim, kernel_size=(2, 2, 1), stride=(2, 2, 1))

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        return x
    
