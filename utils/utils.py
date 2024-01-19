#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/12/08 13:25:56
@Author  :   Bo 
'''
import torch 
import random 
import numpy as np 
import os 


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def create_single_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        

def get_dir_name(conf):
    dir_name = "%s_lr_%.3f_version_%d_n_clients_%d_arch_%s" % (
        conf.partition_type, conf.lr, conf.version, conf.n_clients, conf.arch)
    folder_name = "sers_fl_%s" %  conf.aggregation
    conf.folder_name = folder_name 
    conf.dir_name = dir_name
    return conf    


def get_replace_for_init_path(loc):
    if loc == "nobackup":
        rep = "/nobackup/blia/"
    elif loc == "scratch":
        rep = "/scratch/blia/"
    else:
        rep = "../"
    return rep 


def get_path_init(loc):
    if loc == "nobackup":
        return "/nobackup/blia/exp_data/"
    elif loc == "home":
        return "../exp_data/"
    elif loc == "scratch":
        return "/scratch/blia/exp_data/"







    
