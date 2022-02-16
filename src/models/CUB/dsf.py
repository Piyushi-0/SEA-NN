

import numpy as np
import copy, pickle, os, joblib, sys, datetime, random, yaml

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
#plt.rcParams['axes.labelweight'] = 'bold'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

from utils import pre_process
from utils import evaluate
import submod

import logging
from itertools import product

from tqdm import tqdm
import argparse

def createLogHandler(job_name,log_file):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s; , %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

CONFIG_PATH = os.path.join("..", "..", "..", "config")
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config
config_fname = "config_cub.yaml"
config = load_config(config_fname)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#I_ind = config['methods']
sp_h = config['sp_h'] #height of the window 
sp_w = config['sp_w'] #width of the window  

w = config['w']
h = config['h']
#epochs = config['epochs']#100
init_wt = config['init_wt']
delta = config['delta']#1e-5
#wt_reg = config['wt_reg']#1e-3
#thresholds = config['thresholds']

I_ind = ["ig", "vg", "sg3"] #


epochs = 50
thresholds = [5,10,15,20,25,30,35,40,45,50]

all_paths = [os.path.join(os.environ['HOME'],'data','CUB_200_2011', i) for i in config["all_paths"]]
device_I = 'cuda'
cmap = 'hot'
new_h = int(h/sp_h) #height of sub-sampled image
new_w = int(w/sp_w) #width of sub-sampled image
sq_n_sb_px = new_h # =new_h# square-root no. of sub-pixels

# helper codes
def get_clf(device):
    model = models.resnet18(pretrained=True)
    last_layer_input = model.fc.in_features
    replaced_last_layer = nn.Linear(in_features=last_layer_input, out_features=200, bias=True)
    model.fc = replaced_last_layer

    ckpt = torch.load(os.path.join("..","..","..","models","CUB","classifier-ckpt.pth"))
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    return model

def get_transformed_img(data_path, image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(w),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def get_inp(data_path, image, device):
    transformed_img = get_transformed_img(data_path, image)

    input = transformed_img.unsqueeze(0)
    input = input.to(device)
    return input

def get_paths(hp_index, img_name):
    for p in [os.path.join("..", "..", "..", "models", "CUB" , "DSFs", str(hp_index)),     os.path.join("..", "..", "..", "logs", "CUB" , str(hp_index)),     os.path.join("..","..","..","logs","CUB",str(hp_index),"curves")]:
        if not os.path.exists(p):
            os.makedirs(p)
        map_path = os.path.join("..", "..", "..", "logs", "CUB", str(hp_index), "{}.pickle".format(img_name))
        dsf_path = os.path.join("..", "..", "..", "models", "CUB" , "DSFs", str(hp_index), "{}.pt".format(img_name))
        curve_path = os.path.join("..","..","..","logs","CUB",str(hp_index),"curves")
    return map_path, dsf_path, curve_path

#code for dsf
def sqrt(input):
    return torch.sqrt(input)

class DSF(nn.Module):
    def __init__(self):
        super(DSF, self).__init__()
        self.fc1 = nn.Linear(sq_n_sb_px * sq_n_sb_px, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = x.view(-1, sq_n_sb_px * sq_n_sb_px)
        x = self.fc1(x)
        x = sqrt(x)
        x = self.fc2(x)
        x = sqrt(x)
        x = self.fc3(x)
        x = sqrt(x)
        x = self.fc4(x)
        return x

class clamp_zero(object):
    def __init__(self):
        pass

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.copy_(torch.clamp(w, min=0))
        if hasattr(module, 'bias'):
            w = module.bias.data
            w.copy_(torch.clamp(w, min=0))

def get_DSF(device, init_wt):
    f = DSF() 

    for W in f.parameters(): 
        if isinstance(init_wt, list):
            W.data.uniform_(init_wt[0], init_wt[1])
        else:
            W.data.uniform_(init_wt, init_wt)

    f = f.to(device)
    f = f.double()
    return f

def get_gtclass(path):
    dix = path.index('.')
    gt_class = int(path[dix-3:dix])
    return gt_class

def train_dsf(model, cfg, hp_index,start,end):
    total_aupc = {"ig":0,"vg":0, "sg":0, "sg3":0, "sea":0, "avg":0}
    ld1 = cfg['ld1']; ld2 = cfg['ld2']; wt_reg = cfg['wt_reg']
    print(f'ld1:{ld1}, ld2:{ld2}, wt_reg:{wt_reg}')
    torch.manual_seed(0); np.random.seed(0)
    clipper = clamp_zero()
    index = 0
    for data_path in all_paths[start:end]:
        for folder in tqdm(os.listdir(data_path)):
            if(str(folder)=='.ipynb_checkpoints'):
                continue
            skip_img = False
            img_name = str(folder).split('.')[0]
            map_path, dsf_path,curve_path = get_paths(hp_index, img_name)
            
            stime = datetime.datetime.now()
            image = Image.open(os.path.join(data_path, folder))

            try:
                inp = get_inp(data_path, image, device)
            except:
                continue
                
            target_class, I_ALL, top_5, input_score = pre_process.get_predic_and_maps(model, inp, device_I, I_ind, device)

            gt_class = get_gtclass(data_path)
            if gt_class not in top_5[:1]:
                continue

            avg = np.mean(I_ALL, axis = 0)
            avg = (avg-avg.min())/(avg.max()-avg.min())

            # hard-thresholded maps
            ht = pre_process.final_ht_proc(sp_w, sp_h, thresholds, I_ALL)
            # sub-sampled hard-thresholded maps
            sub_h = pre_process.final_subI_proc(sp_w, sp_h, I_ALL)

            f = get_DSF(device, init_wt)
            optimizer = torch.optim.Adagrad(f.parameters(), lr_decay = 0.1, weight_decay = wt_reg) #torch.optim.Adam(f.parameters(), weight_decay = wt_reg)
            f.train()
            #loss_plt=[]; loss1_plt = []; loss2_plt = []   
            for epoch in range(epochs):
                loss_1 = None; loss_2 = None
                Adic = submod.c_sb_mx(f, list(ht.keys()), sq_n_sb_px, device)#all A*'s at once
                if isinstance(Adic,torch.Tensor):
                    skip_img = True
                    break
                ASList = list(Adic.values())
                AList_f = f(torch.stack(ASList).double().view(len(ASList), 1, sq_n_sb_px, sq_n_sb_px).to(device))
                tensor_ht = {}

                for xk, k in enumerate(ht):
                    tensor_ht[k] = [torch.Tensor(ht) for ht in ht[k]]
                    all_S_f = f(torch.stack(tensor_ht[k]).double().view(len(tensor_ht[k]), 1, sq_n_sb_px, sq_n_sb_px).to(device))
                    for xs, _ in enumerate(tensor_ht[k]):
                        to_add = AList_f[xk]-all_S_f[xs]+delta
                        if to_add>0:
                            if loss_1 is None:
                                loss_1 = to_add
                            else:
                                loss_1 = loss_1+to_add
                ones_f = f(torch.ones(sq_n_sb_px*sq_n_sb_px).double().view(1, 1, sq_n_sb_px, sq_n_sb_px).to(device))
                tensor_sub_h = [torch.Tensor(s_h) for s_h in sub_h]
                sub_h_f = f(torch.stack(tensor_sub_h).double().view(len(tensor_sub_h), 1, sq_n_sb_px, sq_n_sb_px).to(device))
                for xs_h, _ in enumerate(tensor_sub_h):
                    to_also_add = ones_f-sub_h_f[xs_h]
                    if to_also_add>0:
                        if loss_2 is None:
                            loss_2 = to_also_add
                        else:
                            loss_2 = loss_2+to_also_add
                loss = None
                if loss_1 is not None:
                    loss = ld1*loss_1 
                if loss_2 is not None:
                    if loss is not None:
                        loss = loss+ld2*loss_2 
                    else:
                        loss = ld2*loss_2
                if loss is None:
                    break
                f.zero_grad()
                loss.backward()
                optimizer.step()
                f.apply(clipper)

            if skip_img:
                continue

            index += 1
            sea_attr = submod.sea_nn(f, sq_n_sb_px, sp_w, sp_h, device)
            torch.save(f, dsf_path)
            torch.save(sea_attr, map_path)
            #display_stats_map(data_path, image, loss_plt, I_ind, I_ALL, sea_attr)
            all_maps = {"avg": avg, 'sea' : sea_attr}
            for i in range(len(I_ind)):
                all_maps[I_ind[i]] = I_ALL[i]

            all_curves = {'aupc':{}, 'atv':{}}; all_scores = {'aupc':{}, 'atv':{}}
            for meth in all_maps:
                all_scores['aupc'][meth], all_curves['aupc'][meth] = evaluate.compute_aupc(all_maps[meth], inp, model, 100, target_class, input_score, kernel_size = 16, draw_mode = 0, num_regions = 28, num_perturbs = 1, pert_upto=8)

            """plt.clf()
            for meth in all_maps:
                plt.plot(all_curves['aupc'][meth], label = meth)
            plt.legend()
            plt.show()"""

            for meth in all_maps:
                if meth not in total_aupc:
                    total_aupc[meth] = all_scores['aupc'][meth]
                else:
                    total_aupc[meth] += all_scores['aupc'][meth]
                with open(os.path.join(curve_path,'{}_{}.npy'.format(img_name,meth)),'wb') as f:
                    np.save(f,np.array(all_curves['aupc'][meth]))
            print(total_aupc)
            if index%50==1:
                logger.info('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(hp_index, ld1, ld2, wt_reg, total_aupc["ig"], total_aupc["vg"], total_aupc["sg"], total_aupc["sg3"], total_aupc['avg'], total_aupc['sea'])) 
    return total_aupc

if __name__=="__main__":
    start = 0
    end = 200
    hp_index = "draw_0_hpc_ks16"
    logger = createLogHandler('_', os.path.join("..", "..", "..", "logs", "CUB", "auc_log.csv"))
    logger.info('HP_INDEX, ld1, ld2, wt_reg, AUPC_ig, AUPC_vg, AUPC_sg, AUPC_sg3, AUPC_avg, AUPC_sea')
    parameters = dict(
        ld1 = [0.1],
        ld2 = [10],
        wt_reg = [1e-6]
        )
    param_values = [v for v in parameters.values()]
    model = get_clf(device)

    for ld1, ld2, wt_reg in product(*param_values): 
        print(f'HP Index {hp_index}, Start {start}, End {end}')
        cfg = {'ld1': ld1, 'ld2': ld2, 'wt_reg': wt_reg}
        all_aupc = train_dsf(model, cfg, hp_index, start, end) 

        logger.info('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(hp_index, ld1, ld2, wt_reg, all_aupc["ig"], all_aupc["vg"], all_aupc["sg"], all_aupc["sg3"], all_aupc['avg'], all_aupc['sea'])) 
