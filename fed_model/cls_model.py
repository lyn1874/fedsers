#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   cls_model.py
@Time    :   2022/12/08 13:14:53
@Author  :   Bo 
'''
import torch 
import fed_model.vit as vit 
    
    
def get_model(conf, device):
    if "vit" in conf.arch:
        model_obj = vit.ViT()
        model_obj.to(device)
    return model_obj 