# imports
import numpy as np
import copy, pickle, os, joblib, sys, datetime, random, yaml

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['axes.labelweight'] = 'bold'

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

# Decided hyper-parameters
CONFIG_PATH = os.path.join("..", "..", "..", "config")
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config
config_fname = "config_imagenet.yaml"
config = load_config(config_fname)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
I_ind = config['methods']
sp_h = config['sp_h'] #height of the window 
sp_w = config['sp_w'] #width of the window  
w = config['w']
h = config['h']
epochs = config['epochs']#100
init_wt = config['init_wt']
delta = config['delta']#1e-5
#wt_reg = config['wt_reg']#1e-3
thresholds = config['thresholds']
all_paths = [os.path.join(os.environ['HOME'],'data','tiny-imagenet-200', i) for i in config["all_paths"]]
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

    model.load_state_dict(torch.load(os.path.join('..','..','..','models','ImageNet','classifier.pt')))

    return model

def get_transformed_img(data_path, image):
    transform = transforms.Compose([
        transforms.Resize((w,h)),
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
    for p in [os.path.join("..", "..", "..", "models", "ImageNet" , "DSFs", str(hp_index)), os.path.join("..", "..", "..", "models", "ImageNet" , "DSFs", str(hp_index))]:
        if not os.path.exists(p):
            os.makedirs(p)
        map_path = os.path.join("..", "..", "..", "logs", "ImageNet", str(hp_index), "{}.pickle".format(img_name))
        dsf_path = os.path.join("..", "..", "..", "models", "ImageNet" , "DSFs", str(hp_index), "{}.pt".format(img_name))
        curve_path = os.path.join("..","..","..","logs","ImageNet",str(hp_index),"curves")
    return map_path, dsf_path, curve_path

#code for dsf
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
#         x = F.sigmoid(x)
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

def train_dsf(model, cfg, hp_index,start,end):
    total_aupc = {}
    ld1 = cfg['ld1']; ld2 = cfg['ld2']; wt_reg = cfg['wt_reg']
    print(f'ld1:{ld1}, ld2:{ld2}, wt_reg:{wt_reg}')
    torch.manual_seed(0); np.random.seed(0)
    clipper = clamp_zero()
    for data_path in all_paths[start:end]:
        for folder in tqdm(os.listdir(data_path)):
            if(str(folder)=='.ipynb_checkpoints'):
                continue
            skip_img = False
            img_name = str(folder).split('.')[0]
            map_path, dsf_path,curve_path = get_paths(hp_index, img_name)
            print(img_name)
            stime = datetime.datetime.now()
            image = Image.open(os.path.join(data_path, folder))

            try:
                inp = get_inp(data_path, image, device)
            except:
                continue
            target_class, I_ALL = pre_process.get_predic_and_maps(model, inp, device_I, I_ind, device)

            # hard-thresholded maps
            ht = pre_process.final_ht_proc(sp_w, sp_h, thresholds, I_ALL)
            # sub-sampled hard-thresholded maps
            sub_h = pre_process.final_subI_proc(sp_w, sp_h, I_ALL)

            f = get_DSF(device, init_wt)
            optimizer = torch.optim.Adagrad(f.parameters(), lr_decay = 0.1, weight_decay = wt_reg) #torch.optim.Adam(f.parameters(), weight_decay = wt_reg)
            f.train()
            loss_plt=[]; loss1_plt = []; loss2_plt = []    
            for epoch in range(epochs):
                loss_1 = None; loss_2 = None
                Adic = submod.c_sb_mx(f, list(ht.keys()), sq_n_sb_px, device)#all A*'s at once
                if isinstance(Adic,torch.Tensor):
                    skip_img = True
                    break
                ASList = list(Adic.values())
                AList_f = f(torch.stack(ASList).double().view(len(ASList), 1, sq_n_sb_px, sq_n_sb_px).to(device))
                tensor_ht = {}
                # loss_1: loss with hard thresholded maps sub-sampled
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
                # loss_2: loss with original attribution maps
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
            
            if skip_img:
                continue
				
            I_ind.append('avg')
            num_maps = len(I_ALL)
            avg_map = np.zeros_like(I_ALL[0])
            for attr_map in I_ALL:
                avg_map += attr_map
            avg_map = avg_map/num_maps
            I_ALL.append(avg_map)
			
            sea_attr = submod.sea_nn(f, sq_n_sb_px, sp_w, sp_h, device)
            torch.save(f, dsf_path)
            torch.save(sea_attr, map_path)
            #display_stats_map(data_path, image, loss_plt, I_ind, I_ALL, sea_attr)
            all_maps = {I_ind[0] : I_ALL[0], I_ind[1] : I_ALL[1], I_ind[2] : I_ALL[2], I_ind[3]: I_ALL[3], 'sea' : sea_attr}
            all_scores,all_curves = evaluate.AUC(model, inp, all_maps, device, target_class)
            for meth in all_maps:
                if meth not in total_aupc:
                    total_aupc[meth] = all_scores[meth]['aupc']
                else:
                    total_aupc[meth] += all_scores[meth]['aupc']
                with open(os.path.join(curve_path,'{}_{}.npy'.format(img_name,meth)),'wb') as f:
                    np.save(f,np.array(all_curves[meth]))
            
    return total_aupc

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',dest='start',type=int,default=None)
    parser.add_argument('--end',dest='end',type=int,default=None)
    parser.add_argument('--hpindex',dest='hpindex',type=int,default=None)
    args = parser.parse_args()
    logger = createLogHandler('_', os.path.join("..", "..", "..", "logs", "ImageNet", "auc_log.csv"))
    logger.info('HP_INDEX, ld1, ld2, wt_reg, AUPC_{}, AUPC_{}, AUPC_{}, AUPC_avg, AUPC_sea'.format(I_ind[0], I_ind[1], I_ind[2]))
    parameters = dict(
        ld1 = [0.1],
        ld2 = [0.1],
        wt_reg = [1e-3]
    )
    param_values = [v for v in parameters.values()]
    model = get_clf(device)
    
    if args.hpindex!=None:
        hp_index = args.hpindex
    else:
        hp_index = 0
        
    if(args.start==None or args.end==None):
        print('Start and End need to be specified')
        sys.exit()
        
    for ld1, ld2, wt_reg in product(*param_values): 
        print(f'HP Index {hp_index}')
        cfg = {'ld1': ld1, 'ld2': ld2, 'wt_reg': wt_reg}
        all_aupc = train_dsf(model, cfg, hp_index,args.start,args.end) 
        logger.info('{}, {}, {}, {}, {:.2E}, {:.2E}, {:.2E}, {:.2E}, {:.2E}'.format(hp_index, ld1, ld2, wt_reg, all_aupc[I_ind[0]], all_aupc[I_ind[1]], all_aupc[I_ind[2]], all_aupc[I_ind[3]],all_aupc['sea'])) 
        hp_index += 1
