#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   create_dataset.py
@Time    :   2022/09/12 09:18:15
@Author  :   Bo 
'''
import numpy as np 
import torch
import os
import pickle 
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def load_data_batch(_input, _target, is_on_cuda=True):
    """Args:
    conf: argparse
    _input: [batch_size, channel, imh, imw]
    _target: [batch_size]
    """
    if is_on_cuda == True:
        _input, _target = _input.cuda(), _target.cuda()
    _data_batch = {"input": _input, "target": _target}
    return _data_batch


class SERSMap(Dataset):
    def __init__(self, maps, labels, transform):
        super().__init__()
        self.maps = maps
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, index):
        s_map = self.maps[index]
        s_label = self.labels[index]
        if self.transform is not None:
            s_map = self.transform(s_map)
        return s_map, s_label


def define_data_loader(data_to_load, batch_size, shuffle=True, num_workers=2):
    """Args: 
    conf: the argparse 
    dataset: a dictionary or i.e., train_dataset, val_dataset from the define_dataset function
    localdata_id: client id
    is_train: bool variable 
    shuffle: bool variable 
    data_partitioner: a class: pp.DataPartitioner
    Ops:
    during training, conf.user_original_client_data is always "combine". However, during testing, it
    can be set as "only_real" or "only_fake" for evaluating the gradients
    """
    data_loader = torch.utils.data.DataLoader(data_to_load, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=num_workers, 
                                            pin_memory=True,
                                            drop_last=True)
    return data_loader


def get_tr_data(client_index, batch_size, partition_type):
    if partition_type == "iid" or partition_type == "non_iid":
        path2load = "../rs_dataset/simulate_sers_maps/%s/client_%02d.npz" % (partition_type, client_index)
        tr_data = np.load(path2load)
        tr_im, tr_la = tr_data["arr_0"].astype(np.float32), tr_data["arr_1"].astype(np.int64)

    elif partition_type == "centralised":
        path2load = "../rs_dataset/simulate_sers_maps/non_iid/"
        tr_im, tr_la = [], []
        sub_dir = [v for v in os.listdir(path2load) if "client_" in v]
        for i in np.arange(len(sub_dir)):
            _im = np.load(path2load + "/client_%02d.npz" % i)
            tr_im.append(_im["arr_0"])
            tr_la.append(_im["arr_1"])
        tr_im = np.concatenate(tr_im, axis=0).astype(np.float32)
        tr_la = np.concatenate(tr_la, axis=0).astype(np.int64)                
        
    tr_dataset = SERSMap(tr_im, tr_la, transforms.ToTensor())
    tr_loader = define_data_loader(tr_dataset, batch_size, True)
    return tr_loader 


def get_test_dataset(batch_size):
    tt_path = "../rs_dataset/simulate_sers_maps/test_data.npz"
    tt_data = np.load(tt_path)
    tt_map, tt_la = tt_data["arr_0"].astype(np.float32), tt_data["arr_1"].astype(np.int64)
    tt_transform = transforms.ToTensor()

    tt_dataset = SERSMap(tt_map, tt_la, tt_transform)
    tt_loader = torch.utils.data.DataLoader(tt_dataset, batch_size=batch_size,
                                            shuffle=False, drop_last=False)
    return tt_loader 


def split_data(content, split, num_tr, tds_dir, save=False, show=False):
    np.random.seed(1024)
    
    key_use = sorted([v for v in list(content.keys()) if "Map_" in v])
    
    client_data_group = [content[v] for v in key_use[1:]]
    peak_group = [content[v.replace("Map_", "Peak_")] for v in key_use[1:]]
    
    tr_client_data = [v[:num_tr] for v in client_data_group]
    tr_conc = [np.repeat([float(v.split("_")[1])], num_tr) for v in key_use[1:]]
    tr_peak = [v[:num_tr] for v in peak_group]
    
    tt_client_data = [v[num_tr:] for v in client_data_group]
    base_data = content[key_use[0]]
    
    imh, imw, ch = np.shape(base_data)[1:]
    
    base_split = np.reshape(base_data, [len(key_use) - 1, -1, imh, imw, ch])
    tt_client_data += [v[num_tr:] for v in base_split]
    tr_base_split = [v[:num_tr] for v in base_split]
    tr_base_peak = content[key_use[0].replace("Map_", "Peak_")]
    tr_base_peak = np.reshape(tr_base_peak, [len(key_use)-1, -1, np.shape(tr_base_peak)[-1]])[:, :num_tr]
    print('The shape of the tr peak', np.shape(tr_peak))
    print("The shape of the baseline tr peak", np.shape(tr_base_peak))

    print("The shape of the test data")
    for i, v in enumerate(tt_client_data):
        if i < len(key_use)-1:
            print(key_use[i+1], np.shape(v))
        else:
            print(key_use[0], np.shape(v))

    tt_data = np.concatenate(tt_client_data, axis=0)
    tt_label = np.zeros([len(tt_data)])
    tt_label[:len(tt_label) // 2] = 1.0
    
    print("There are %d number of test SERS maps" % len(tt_data))
    print("The distribution of the tt label", np.unique(tt_label, return_counts=True))
    if save:
        if not os.path.exists("../rs_dataset/simulate_sers_maps/test_data.npz"):
            np.savez("../rs_dataset/simulate_sers_maps/" + '/test_data', tt_data, tt_label)
    m, b = len(tr_client_data[0]), np.shape(tr_base_split[0])[0]
    
    if split == "non_iid":
        client_data_group = [np.concatenate([v, tr_base_split[i]]) for i, v in enumerate(tr_client_data)]
        label_group = [np.concatenate([np.ones([m]), np.zeros([b])], axis=0) for _ in key_use[1:]]
        tr_peak_group = [np.concatenate([v, q], axis=0) for v, q in zip(tr_peak, tr_base_peak[:len(tr_peak)])]
    else:
        client_data_group = np.concatenate(tr_client_data, axis=0)
        shuffle_index = np.random.choice(np.arange(len(client_data_group)), len(client_data_group), 
                                        replace=False)
        
        tr_conc = np.reshape(np.concatenate(tr_conc, axis=0)[shuffle_index], [len(key_use)-1, -1])
        peak_group = np.reshape(np.concatenate(tr_peak, axis=0)[shuffle_index], [len(key_use)-1, -1, np.shape(tr_peak)[-1]])
        client_data_group = np.reshape(client_data_group[shuffle_index], [len(key_use)-1, -1, imh, imw, ch])
        client_data_group = [np.concatenate([v, tr_base_split[i]]) for i, v in enumerate(client_data_group)]        
        label_group = [np.concatenate([np.ones([m]), np.zeros([b])], axis=0) for _ in key_use[1:]]
        tr_peak_group = [np.concatenate([v, q], axis=0) for v, q in zip(peak_group, tr_base_peak[:len(peak_group)])]
    print("There are %d clients" % len(client_data_group))
    for i, s_stat in enumerate(client_data_group):
        print("Client-%02d: sers map shape" % i, np.shape(s_stat), np.unique(label_group[i], return_counts=True), np.unique(tr_conc[i], return_counts=True))
    if save:
        if not os.path.exists(tds_dir):
            os.makedirs(tds_dir)
        for i in range(len(client_data_group)):
            np.savez(tds_dir+"/client_%02d" % i, client_data_group[i], label_group[i])
    if show:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        for i, s_value in enumerate(client_data_group):
            rand_index = list(np.random.choice(np.where(label_group[i] == 0)[0], 6, replace=False))
            rand_index += list(np.random.choice(np.where(label_group[i] == 1)[0], 6, replace=False))
            
            for j, rand_sers in enumerate(rand_index):
                ax = fig.add_subplot(len(client_data_group), 12, i * 12 + j + 1)
                peak_use = tr_peak_group[i][rand_sers]
                ax.imshow(np.sum(np.array(s_value[rand_sers])[:, :, peak_use.astype(np.int32)], axis=-1))
                ax.set_title(label_group[i][rand_sers])
                ax.set_xticks([])
                ax.set_yticks([])



def save_data_to_clients(iid_or_non_iid, num_tr, save=False, show=False):
    file = pickle.load(open("../rs_dataset/simulate_sers_maps/Type_2_ar_bg_with_concentration_seed_1002.obj", "rb"))
    print("Saving the data %s with %d training data per client" % (iid_or_non_iid, num_tr))
    split_data(file, iid_or_non_iid, num_tr, tds_dir="../rs_dataset/simulate_sers_maps/%s/" % iid_or_non_iid, save=save, show=show)
