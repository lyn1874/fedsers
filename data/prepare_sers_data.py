#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   prepare_sers_data.py
@Time    :   2023/12/11 17:33:08
@Author  :   Bo 
'''
import numpy as np
from data.generate_data import SERSGeneratorMore
import matplotlib.pyplot as plt
import pickle
import os


def check_signaltobackground_ratio(concentration, num_hotspot, hotspot_scale=3):
    sbr = 1 / 100 * (np.exp(concentration) - 1)
    seed = 1002 + np.random.randint(0, 10**8, 1)[0]
    Nw = 110 
    num_peak = 2 
    np.random.seed(seed)
    c = np.array([np.random.uniform(20, Nw / 2 - 7), 
                  np.random.uniform(Nw / 2+7, Nw - 20)])
    gamma = np.random.normal(5, 1, size=num_peak)
    eta = np.random.rand(num_peak)

    data_obj = SERSGeneratorMore([30, 30], num_hotspot, 110, sbr, concentration,  0.2, seed, 
                                 seed, c=c, gamma=gamma, eta=eta, hotspot_size=hotspot_scale)
    x, dd, bb, signal_stat, contaminate_stat = data_obj.forward("ar", [0.1, 0.92])
    print("The min and max of the background", np.min(bb), np.max(bb))
    print("The min and max of the signal", np.min(dd), np.max(dd))
    
    return x, dd, bb, signal_stat, contaminate_stat, c
    
    
class SaveSimulateData(object):
    def __init__(self, mapsize, Nw, num_measurements, concentration, quantification=False, tds_dir=None, save=False, 
                 fix_hotspot_intensity=False, 
                 hotspot_scale=3):
        super(SaveSimulateData, self).__init__()
        self.mapsize = mapsize 
        self.Nw = Nw 
        self.num_measurements = num_measurements
        self.concentration = concentration
        self.sbr = 1 / 100 * (np.exp(self.concentration) - 1)
        self.quantification=quantification 
        self.fix_hotspot_intensity = fix_hotspot_intensity
        print("Fix hotspot intensity", self.fix_hotspot_intensity)
        zero_num_measurements = self.num_measurements * (len(self.concentration) - 1)
        others_num_measurements = self.num_measurements * (len(self.concentration) - 1)
        self.seed = 1002 + np.arange((zero_num_measurements + others_num_measurements) * 2)[::2]
        self.tds_dir = tds_dir 
        self.save = save 
        self.noise_scale = 0.1 
        self.hotspot_scale = hotspot_scale
        self.gamma_avg = 4 
    
    def get_obj(self, num_hotspot, s_sbr, s_conc, seed_use, seed_intensity_use, c, gamma, eta):
        data_obj = SERSGeneratorMore(self.mapsize, num_hotspot, self.Nw, s_sbr, s_conc, 0.2, seed_use, 
                                     seed_intensity_use, c=c, gamma=gamma, eta=eta, 
                                     fix_hotspot_intensity=self.fix_hotspot_intensity,
                                     hotspot_size=self.hotspot_scale)
        return data_obj
        
    def test_type_two_advance_data(self, num_peak=2, bg_method="ar", bg_para=[0.5, 0.92]):
        sers_maps, signal_stat_g = [], [[] for _ in range(7)]
        concentration_g, peaks = [], []
        clean_vogit_group = []
        if self.quantification:
            zero_num_measurements = self.num_measurements
        else:
            zero_num_measurements = self.num_measurements * (len(self.concentration) - 1)
        others_num_measurements = self.num_measurements * (len(self.concentration) - 1)
        hotspot_g = np.array(list(np.ones([zero_num_measurements])) + list(np.random.randint(1, 11, others_num_measurements))).astype(np.int32)
        lower = 0
        for i, s_conc in enumerate(self.concentration):
            sers_map_per_conc, peak_per_conc, vogit_per_conc = [], [], []
            num_measurement_use = [zero_num_measurements if s_conc == 0 else self.num_measurements][0]            
            for j in range(num_measurement_use):
                seed_use = self.seed[lower + j]            
                np.random.seed(seed_use)
                if i == 0 and j == 0:
                    c = np.array([np.random.uniform(20, self.Nw / 2 - 7), 
                                np.random.uniform(self.Nw / 2+7, self.Nw - 20)])
                    gamma = np.random.normal(self.gamma_avg, 1, size=num_peak)
                    eta = np.random.rand(num_peak)
                data_obj = SERSGeneratorMore(self.mapsize, 
                                            hotspot_g[lower + j], self.Nw, self.sbr[i], 
                                            s_conc, self.noise_scale, seed=seed_use,
                                            seed_intensity=self.seed[i],
                                            c=c, gamma=gamma, eta=eta,
                                            fix_hotspot_intensity=self.fix_hotspot_intensity,
                                            hotspot_size=self.hotspot_scale)
                x, _, _, signal_stat, _, vp = data_obj.forward(bg_method, bg_para, [])
                sers_map_per_conc.append(np.reshape(x, [self.mapsize[0], self.mapsize[1], self.Nw]))
                for v, q in zip(signal_stat_g[:-1], signal_stat):
                    v.append(q)
                signal_stat_g[-1].append(hotspot_g[lower + j]) 
                peak_per_conc.append(c)
                vogit_per_conc.append(vp)
            lower += num_measurement_use
            sers_maps.append(sers_map_per_conc)
            concentration_g.append(np.repeat(s_conc, num_measurement_use))
            peaks.append(peak_per_conc)
            clean_vogit_group.append(vogit_per_conc)
        
        print("------------------------------------------------------------------------")
        print("The shape of the sers maps:", np.shape(sers_maps))
        print("The shape and unique of the concentration", np.shape(concentration_g), [np.unique(v) for v in concentration_g])
        print("The shape of the stat", [np.shape(v) for v in signal_stat_g])
        print("The distribution of the number of hotspots", np.unique(np.array(signal_stat_g[-1]), return_counts=True))
        print("The shape of the number of hotspots", np.shape(hotspot_g), np.unique(hotspot_g, return_counts=True))
        print("------------------------------------------------------------------------")
        return sers_maps, concentration_g, peaks, signal_stat_g, [], clean_vogit_group       
    
    def test_type_three_data(self, num_peak=2, bg_method="ar", bg_para=[0.5, 0.92]):
        contaminate_peak = [55]
        contaminate_intensity = np.mean(self.sbr) 
        peaks = []
        sers_maps, concentration_g, signal_stat_g, contaminate_stat_g = [], [], [[] for _ in range(7)], [[] for _ in range(3)]
        if self.quantification:
            zero_num_measurements = self.num_measurements
        else:
            zero_num_measurements = self.num_measurements * (len(self.concentration) - 1)
        others_num_measurements = self.num_measurements * (len(self.concentration) - 1)
        hotspot_g = np.array(list(np.ones([zero_num_measurements])) + list(np.random.randint(1, 6, others_num_measurements))).astype(np.int32)
        generate_contaminate = np.zeros([zero_num_measurements + others_num_measurements])
        lower = 0 
        clean_vogit_group, con_vogit_group = [], []
        for i, s_conc in enumerate(self.concentration):
            sers_map_per_conc = []
            peak_per_conc = []
            vogit_per_conc, con_vogit_per_conc = [], []
            num_measurement_use = [zero_num_measurements if s_conc == 0 else self.num_measurements][0]
            for j in range(num_measurement_use):
                seed_use = self.seed[lower + j]
                np.random.seed(seed_use)
                if i == 0 and j == 0:
                    c = np.array([np.random.uniform(20, self.Nw / 2 - 7), 
                                np.random.uniform(self.Nw / 2+7, self.Nw - 20)])
                    gamma = np.random.normal(self.gamma_avg, 1, size=num_peak)
                    eta = np.random.rand(num_peak)
                    gamma_contaminate = np.random.normal(4, 1, 1)
                    eta_contaminate = np.random.rand(1)
                data_obj = SERSGeneratorMore(self.mapsize, hotspot_g[lower + j], self.Nw, self.sbr[i], 
                                             s_conc, self.noise_scale, seed=seed_use,
                                             seed_intensity=self.seed[i],
                                             c=c, gamma=gamma, eta=eta,
                                             fix_hotspot_intensity=self.fix_hotspot_intensity,
                                             hotspot_size=self.hotspot_scale)

                contaminate = contaminate_intensity + abs(np.random.normal(0.1, 0.1, [1])[0])
                existence_of_contaminate = np.random.random()
                if existence_of_contaminate >= 0.8:
                    contaminate_parameters = [contaminate_peak, gamma_contaminate, eta_contaminate, contaminate]
                    generate_contaminate[lower + j] = 1
                else:
                    contaminate_parameters = []
                x, _, _, signal_stat, contaminate_stat, vp = data_obj.forward(bg_method, bg_para, contaminate_parameters)
                sers_map_per_conc.append(np.reshape(x, [self.mapsize[0], self.mapsize[1], self.Nw]))
                for v, q in zip(signal_stat_g[:-1], signal_stat):
                    v.append(q)
                signal_stat_g[-1].append(hotspot_g[lower+j])
                for v, q in zip(contaminate_stat_g, contaminate_stat):
                    v.append(q)
                peak_per_conc.append(np.array(list(c) + contaminate_peak))
                vogit_per_conc.append(vp)
            lower += num_measurement_use
            peaks.append(peak_per_conc)
            concentration_g.append(np.repeat(s_conc, num_measurement_use))
            sers_maps.append(sers_map_per_conc)
            clean_vogit_group.append(vogit_per_conc)
        # signal_stat_g[-1] = hotspot_g
        print("-----------------------------------------------------------------------------------")
        print("The shape of the concentration", np.shape(concentration_g), [np.unique(v) for v in concentration_g])
        print("The shape of the sers maps", np.shape(sers_maps))
        print("The shape of the signal statistics", [np.shape(v) for v in signal_stat_g])
        print("The shape of the contaminate stat", [np.shape(v) for v in contaminate_stat_g])
        print("The shape of the number of hotspots and the distribution", np.shape(hotspot_g), np.unique(hotspot_g, return_counts=True))
        print("The shape of the actual number of hotspots distribution", np.unique(signal_stat_g[-1], return_counts=True))
        print("The shape of the peaks", [np.shape(v) for v in peaks])
        print("-----------------------------------------------------------------------------------")       
        return sers_maps, concentration_g, peaks, signal_stat_g, [contaminate_stat_g, generate_contaminate], clean_vogit_group
    
    def test_type_four_data(self, num_peak=2, bg_method="ar", bg_para=[0.5, 0.92]):
        sers_maps, concentration_g, signal_stat_g, contaminate_stat_g = [], [], [[] for _ in range(7)], [[] for _ in range(3)]
        if self.quantification:
            zero_num_measurements = self.num_measurements
        else:
            zero_num_measurements = self.num_measurements * (len(self.concentration) - 1)
        other_num_measurements = self.num_measurements * (len(self.concentration) - 1)
        hotspot_g = np.array(list(np.ones([zero_num_measurements])) + list(np.random.randint(1, 6, other_num_measurements))).astype(np.int32)
        tot_measurements = int(zero_num_measurements + other_num_measurements)
        gamma_contaminate = np.random.normal(4, 1, [tot_measurements])
        eta_contaminate = np.random.rand(tot_measurements)
        c_contaminate = np.random.uniform(20, self.Nw-20, [tot_measurements])
        intensity_contaminate = abs(np.random.normal(np.max(self.sbr), np.std(self.sbr[1:]), [tot_measurements])) #+ np.mean(self.sbr) / 2
        generate_contaminate = np.zeros([tot_measurements])
        peaks = []
        lower = 0
        clean_vogit_group = []
        for i, s_conc in enumerate(self.concentration):
            sers_map_per_conc = []
            peak_per_conc = []
            num_measurement_use = [zero_num_measurements if s_conc == 0 else self.num_measurements][0]
            vogit_per_conc, con_vogit_per_conc = [], []
            for j in range(num_measurement_use):
                seed_use = self.seed[lower + j]
                np.random.seed(seed_use)
                if i == 0 and j == 0:
                    c = np.array([np.random.uniform(20, self.Nw / 2 - 7), 
                                np.random.uniform(self.Nw / 2+7, self.Nw - 20)])
                    gamma = np.random.normal(self.gamma_avg, 1, size=num_peak)
                    eta = np.random.rand(num_peak)
                data_obj = SERSGeneratorMore(self.mapsize, hotspot_g[lower + j], self.Nw, self.sbr[i], 
                                             s_conc, self.noise_scale, seed=seed_use,
                                             seed_intensity=self.seed[i],
                                             c=c, gamma=gamma, eta=eta,
                                             fix_hotspot_intensity=self.fix_hotspot_intensity,
                                             hotspot_size=self.hotspot_scale)
                if np.random.random() > 0.5:
                    contaminate_parameter = [[c_contaminate[lower + j]], 
                                             gamma_contaminate[lower + j], 
                                             eta_contaminate[lower + j],
                                             intensity_contaminate[lower + j]]
                    generate_contaminate[lower + j] = 1
                else:
                    contaminate_parameter = []
                x, _, _, signal_stat, contaminate_stat, vp = data_obj.forward(bg_method, bg_para, contaminate_parameter)
                sers_map_per_conc.append(np.reshape(x, [self.mapsize[0], self.mapsize[1], self.Nw]))
                for v, q in zip(signal_stat_g[:-1], signal_stat):
                    v.append(q)
                for v, q in zip(contaminate_stat_g, contaminate_stat):
                    v.append(q)
                peak_per_conc.append(np.array(list(c) + [c_contaminate[lower+ j]]))
                vogit_per_conc.append(vp)
            lower += num_measurement_use
            sers_maps.append(sers_map_per_conc)
            peaks.append(peak_per_conc)
            concentration_g.append(np.repeat(s_conc, num_measurement_use))
            clean_vogit_group.append(vogit_per_conc)
        signal_stat_g[-1] = hotspot_g                
        print("-----------------------------------------------------------------------------------")
        print("The shape of the concentration", np.shape(concentration_g), [np.unique(v) for v in concentration_g])
        print("The shape of the sers maps", np.shape(sers_maps))
        print("The shape of the signal statistics", [np.shape(v) for v in signal_stat_g])
        print("The shape of the contaminate statistics", [np.shape(v) for v in contaminate_stat_g])
        print("The shape of the number of hotspots and the distribution", np.shape(hotspot_g), np.unique(hotspot_g, return_counts=True))
        print("Number of contaminates added", np.sum(generate_contaminate) / len(generate_contaminate))
        print("-----------------------------------------------------------------------------------")       
        return sers_maps, concentration_g, peaks, signal_stat_g, [contaminate_stat_g, gamma_contaminate, eta_contaminate, c_contaminate, intensity_contaminate, generate_contaminate], clean_vogit_group
    
    def save_simulate_data(self, data_type):
        """Generate data with different concentrations, Args:
        num_measurements: the amount of maps that it needs to generate
        bg_method: "ar"
        seed: int
        concentration_level: if it's [], then the default on
        """
        if data_type == "Type_2":
            sers_maps, concentration, peaks, signal_stat_g, contaminate_stat_g, vogit_g = self.test_type_two_advance_data(num_peak=2, bg_method="ar", bg_para=[0.2, 0.92])
        elif data_type == "Type_3":
            sers_maps, concentration, peaks, signal_stat_g, contaminate_stat_g, vogit_g = self.test_type_three_data(num_peak=2, 
                                                                                                           bg_method="ar", bg_para=[0.2, 0.92],
                                                                                                           )
        elif data_type == "Type_4":
            sers_maps, concentration, peaks, signal_stat_g, contaminate_stat_g, vogit_g = self.test_type_four_data(num_peak=2, bg_method="ar",
                                                                                                          bg_para=[0.2, 0.92])
        data_type_name = data_type            

        # produce_sers_maps_figure(sers_maps, concentration, peaks, signal_stat_g, contaminate_stat_g, self.tds_dir, data_type_name, save=self.save)
        if self.save:
            sers_maps_obj = {}
            for i, s_conc in enumerate(sorted([v[0] for v in concentration])):
                sers_maps_obj["Map_%.4f_concentration" % s_conc] = sers_maps[i]
                sers_maps_obj["Peak_%.4f_concentration" % s_conc] = peaks[i]
            sers_maps_obj["statistics"] = [signal_stat_g, vogit_g]
            if data_type == "Type_3" or data_type == "Type_4":
                sers_maps_obj["contaminate_statistics"] = contaminate_stat_g
            print("Saved data name", data_type_name)
            with open(self.tds_dir + "/%s_%s_bg_with_concentration_seed_%d.obj" % (data_type_name, "ar", 1002), "wb") as f:
                pickle.dump(sers_maps_obj, f)
            
            
def save_data(data_type, quantification, save):
    mapsize = [30, 30]
    num_measurements = [500 if not quantification else 200][0]
    
    # num_measurements = [100 if not quantification else 150][0]
    Nw = 110 
    if data_type != "Type_6":
        # concentration = np.array([0, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1] + list(np.linspace(0.2, 0.5, 4)))
        concentration = np.array([0, 0.01, 0.025, 0.05, 0.1] + list(np.linspace(0.2, 0.5, 4)))

    else:
        concentration = np.round(np.linspace(0, 0.5, 12), 2)
    if data_type == "Type_5" or data_type == "Type_6":
        fix_hotspot_intensity = True 
    else:
        fix_hotspot_intensity = False 
    if data_type == "Type_7" or data_type == "Type_8":
        hotspot_scale = 30
    else:
        hotspot_scale = 5
    print("Fixing hotspot intensity", fix_hotspot_intensity)
    # concentration = np.array([0] + [0.05, 0.1, 0.15] + list(np.linspace(0.2, 1.0, 9)))
    data_obj = SaveSimulateData(mapsize, Nw, num_measurements, concentration, quantification=quantification, tds_dir="../rs_dataset/simulate_sers_maps/", save=save,
                                fix_hotspot_intensity=fix_hotspot_intensity, hotspot_scale=hotspot_scale)
    print("------------------------------------------------------------")
    print("Concentration:", concentration)
    print("Data type:", data_type)
    print("Hotspot size", hotspot_scale)
    print("------------------------------------------------------------")
    data_obj.save_simulate_data(data_type)
    
    
def split_data(content, split, tds_dir, save=False):
    key_use = sorted([v for v in list(content.keys()) if "Map_" in v])
    client_data_group = [content[v] for v in key_use[1:]]
    base_data = content[key_use[0]]
    imh, imw, ch = np.shape(base_data)[1:]
    base_split = np.reshape(base_data, [len(key_use) - 1, -1, imh, imw, ch])
    m, b = len(client_data_group[0]), np.shape(base_split)[1]
    if split == "non_iid":
        client_data_group = [np.concatenate([v, base_split[i]]) for i, v in enumerate(client_data_group)]
        label_group = [np.concatenate([np.ones([m]), np.zeros([b])], axis=0) for _ in key_use[1:]]
        if save:
            if not os.path.exists(tds_dir):
                os.makedirs(tds_dir)
            for i in range(10):
                np.save(tds_dir+"/client_%02d" % i, client_data_group[i], label_group[i])
    else:
        client_data_group = np.concatenate(client_data_group[:-1], axis=0)
        shuffle_index = np.random.choice(np.arange(len(client_data_group)), len(client_data_group), 
                                        replace=False)

        client_data_group = np.reshape(client_data_group[shuffle_index], [len(key_use)-2, -1, imh, imw, ch])
        client_data_group = [np.concatenate([v, base_split[i]]) for i, v in enumerate(client_data_group)]
        
        label_group = [np.concatenate([np.ones([m]) + np.zeros([b])], axis=0) for _ in key_use[2:]]
        if save:
            if not os.path.exists(tds_dir):
                os.makedirs(tds_dir)
            for i in range(10):
                np.save(tds_dir+"/client_%02d" % i, client_data_group[i], label_group[i])
    return client_data_group, label_group
    