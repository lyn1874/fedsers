#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   create_train.py
@Time    :   2022/10/10 14:11:23
@Author  :   Bo 
'''
import train_fed_avg as train_fed_avg 
import configs.conf as const 
import utils.utils as utils 
import torch 
import os 


device = torch.device("cuda")


def run(conf):
    if conf.use_local_id == 0 and conf.communication_round == 0:
        conf = utils.get_dir_name(conf)
        if not os.path.exists("../exp_data/%s/%s" % (conf.folder_name, conf.dir_name)):
            os.makedirs("../exp_data/%s/%s" % (conf.folder_name, conf.dir_name))
        if not os.path.exists("/nobackup/blia/exp_data/%s/%s/" % (conf.folder_name, conf.dir_name)):
            os.makedirs("/nobackup/blia/exp_data/%s/%s/" % (conf.folder_name, conf.dir_name))
    if conf.aggregation == "fed_avg":
        train_fed_avg.train_with_conf(conf)
        
if __name__ == "__main__":
    a = torch.zeros([1]).to(device)
    conf = const.give_fed_args()
    
    run(conf)