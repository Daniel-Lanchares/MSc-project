# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:18:47 2023

@author: danie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict


from .net_blocks import ResNetBasicBlock

'''
This function should create a XResNet
'''
def create_net(input_channels:int, output_length:int, 
               block=ResNetBasicBlock, deepths=[2, 2, 2, 2], *args, **kwargs):
    '''
    
    Creates the ResNet architecture. ResNet-18 by default

    Parameters
    ----------
    input_channels : int
        Channels of the image (Number of detectors).
    output_length : int
        Number of parameters to do the regresion on.
        Might change meaning when hooking it to the flow
    block : object, optional
        Block type for the net. The default is ResNetBasicBlock.
    deepths : list[int], optional
        Number of blocks in each layer. The default is [2, 2, 2, 2].

    Returns
    -------
    object
        The resisual network.

    '''
    return ResNet(input_channels, output_length, block=block, deepths=deepths, *args, **kwargs)

class ResNet(nn.Module):
    '''
    The ResNet is composed of an encoder and a decoder
    '''
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by subsequent layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1)) #TODO substitute with blurpool
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

