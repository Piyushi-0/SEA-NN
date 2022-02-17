import numpy as np
import copy, pickle, os, joblib, sys, datetime, random, yaml
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['axes.labelweight'] = 'bold'

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from utils import pre_process
from utils import evaluate
import submod

import logging
from itertools import product

import argparse

I_ind = ["ig", "vg", "sg3"]
epochs = 50
thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

if not os.path.exists(os.path.join("..", "..", "..", "logs", "FMNIST")):
    os.makedirs(os.path.join("..", "..", "..", "logs", "FMNIST"))

def createLogHandler(job_name,log_file):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s; , %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Decided hyper-parameters
CONFIG_PATH = os.path.join("..", "..", "..", "config")
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config
config_fname = "config_fmnist.yaml"
config = load_config(config_fname)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp_h = 1
sp_w = 1
w = 28
h = 28
init_wt = config['init_wt']
delta = config['delta']

device_I = 'cuda'
cmap = 'hot'

new_h = int(h/sp_h)
new_w = int(w/sp_w)

sq_n_sb_px = new_h

def get_clf(device):
    model = torch.load(os.path.join("..", "..", "..", "models", "FMNIST", "classifier.pt"))
    model = model.to(device)
    return model

def get_paths(hp_index, img_name):
    for p in [os.path.join("..", "..", "..", "models", "FMNIST" , "DSFs", str(hp_index)), \
    os.path.join("..", "..", "..", "logs", "FMNIST" , str(hp_index)), \
    os.path.join("..","..","..","logs","FMNIST",str(hp_index),"curves")]:
        if not os.path.exists(p):
            os.makedirs(p)
        map_path = os.path.join("..", "..", "..", "logs", "FMNIST", str(hp_index), "{}.pickle".format(img_name))
        dsf_path = os.path.join("..", "..", "..", "models", "FMNIST" , "DSFs", str(hp_index), "{}.pt".format(img_name))
        curve_path = os.path.join("..","..","..","logs","FMNIST",str(hp_index),"curves")
    return map_path, dsf_path, curve_path

def sqrt(input):
    return torch.sqrt(input)

class DSF(nn.Module):
    def __init__(self):
        super(DSF, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
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

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""

    def __init__(self, data, transform = None):
        """Method to initilaize variables.""" 
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label = []
        image = []

        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

def get_test_loader():
    test_csv = pd.read_csv("/home/piyushi/fmnist/fashion-mnist_test.csv")
    test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(test_set, batch_size=1)
    return test_loader
test_loader = get_test_loader()

def train_dsf(model, cfg, hp_index):
    total_aupc = {"ig":0,"vg":0, "sg":0, "sg3":0, "sea":0, "avg":0}
    ld1 = cfg['ld1']; ld2 = cfg['ld2']; wt_reg = cfg['wt_reg']

    model = get_clf("cuda")
    torch.manual_seed(0); np.random.seed(0)
    clipper = clamp_zero()
    index = 0

    for inp, gt_cls in test_loader:
        if index not in range(4261,7001):
            index+=1
            continue

        skip_img = False
        img_name = "index_{}".format(index)
        map_path, dsf_path, curve_path = get_paths(hp_index, img_name)
        
        target_class, I_ALL, input_score = pre_process.get_predic_and_maps(model, inp, device_I, I_ind, device)
        
        if gt_cls != target_class:
            index+=1
            continue
        
        avg = np.mean(I_ALL, axis = 0)
        avg = (avg-avg.min())/(avg.max()-avg.min())

        ht = pre_process.vanilla_final_ht_proc(thresholds, I_ALL)

        sub_h = I_ALL[:]

        I_ALL.append(avg)
        f = get_DSF(device, init_wt)
        optimizer = torch.optim.Adagrad(f.parameters(), lr_decay = 0.1, weight_decay = wt_reg) #torch.optim.Adam(f.parameters(), weight_decay = wt_reg)

        f.train()
        loss_plt=[]; loss1_plt = []; loss2_plt = []    
        for epoch in range(epochs):
            loss_1 = None; loss_2 = None
            Adic = submod.c_sb_mx(f, list(ht.keys()), sq_n_sb_px, device)#all A*'s at once
            if(isinstance(Adic,torch.Tensor)):
                skip_img = True
                break
            ASList = list(Adic.values())
            AList_f = f(torch.stack(ASList).double().view(len(ASList), 1, sq_n_sb_px, sq_n_sb_px).to(device))
            tensor_ht = {}
            for xk, k in enumerate(ht):
                tensor_ht[k] = [torch.Tensor(ht) for ht in ht[k]] # hard thresholded sub-sampled maps(of all methods) having cardinality k
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

            loss_plt.append(loss.item())
            f.zero_grad()
            loss.backward()
            optimizer.step()
            f.apply(clipper)

        if(skip_img):
            index += 1
            continue

        sea_attr = submod.sea_nn(f, sq_n_sb_px, sp_w, sp_h, device)

        torch.save(f, dsf_path)
        torch.save(sea_attr, map_path)
        all_maps = {"avg": avg, 'sea' : sea_attr}
        for i in range(len(I_ind)):
            all_maps[I_ind[i]] = I_ALL[i]

        all_curves = {'aupc':{}, 'atv':{}}; all_scores = {'aupc':{}, 'atv':{}}
        for meth in all_maps:
            all_scores['aupc'][meth], all_curves['aupc'][meth] = evaluate.compute_aupc(all_maps[meth], inp, model, 100, target_class, input_score, kernel_size = 8, draw_mode = 0, num_regions = 28, num_perturbs = 1, pert_upto=8, FMNIST = True)

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
        index+=1
    return total_aupc


if __name__=="__main__":
    logger = createLogHandler('_', os.path.join("..", "..", "..", "logs", "FMNIST", "tuning_auc_log.csv"))
    #logger.info('HP_INDEX, ld1, ld2, wt_reg, AUPC_{}, AUPC_avg, AUPC_sea'.format(I_ind[2]))
    parameters = dict(
        ld1 = [10],
        ld2 = [10],
        wt_reg = [1e-6]
    )

    param_values = [v for v in parameters.values()]
    model = get_clf(device)
    hp_index = '4_F16'
    for ld1, ld2, wt_reg in product(*param_values):
        print(f'HP Index {hp_index}')
        cfg = {'ld1': ld1, 'ld2': ld2, 'wt_reg': wt_reg}
        all_aupc = train_dsf(model, cfg, hp_index) 
        logger.info('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(hp_index, ld1, ld2, wt_reg, all_aupc["ig"], all_aupc["vg"], all_aupc["sg"], all_aupc["sg3"], all_aupc['avg'], all_aupc['sea']))  
