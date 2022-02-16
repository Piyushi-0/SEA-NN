from captum.attr import GuidedGradCam
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import Deconvolution
from captum.attr import GuidedBackprop
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import DeepLiftShap
from captum.attr import GradientShap
from captum.attr import DeepLiftShap
from captum.attr import InputXGradient

import numpy as np
import torch
import torch.nn.functional as F
import copy

def get_inp(data_path, image, device):
    if "test" in data_path:
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                 transforms.Resize((w,h))])
    else:
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                  transforms.Resize((w,h)),
                                  transforms.ColorJitter(brightness=1, contrast=1, saturation=1)])
    transformed_img = transform(image)
    #tmp_transform_resize = transforms.Compose([transforms.Resize((w, h))])
    transform = transforms.Compose([transforms.ToTensor()])
    transformed_img = transform(transformed_img)

    input = transformed_img.unsqueeze(0)
    input = input.to(device)
    return input

def get_ig(model, inp, target_class):
    '''
    Returns Integrated Gradients attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
            Returns:
                    array   
    '''
    ig = IntegratedGradients(model)
    return ig.attribute(inp, target=target_class)

def get_sg(model, inp, target_class, n_sg):
    '''
    Returns Smooth-Grad attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
                    n_sg (int): no. of samples for smooth-grad function
            Returns:
                    array   
    '''
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    try:
        return nt.attribute(inp, nt_type='smoothgrad', nt_samples=n_sg, target=target_class)
    except:
        return nt.attribute(inp, nt_type='smoothgrad', n_samples=n_sg, target=target_class)

def get_svg(model, inp, target_class, n_sg):
    '''
    Returns Smooth-Grad attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
                    n_sg (int): no. of samples for smooth-grad function
            Returns:
                    array   
    '''
    vg = Saliency(model)
    nt = NoiseTunnel(vg)
    try:
        return nt.attribute(inp, nt_type='smoothgrad', nt_samples=n_sg, target=target_class)
    except:
        return nt.attribute(inp, nt_type='smoothgrad', n_samples=n_sg, target=target_class)

def get_vg(model, inp, target_class):
    '''
    Returns Vanilla-Gradient attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
            Returns:
                    array
    '''
    return Saliency(model).attribute(inp, target=target_class)

def get_dl(model, inp, target_class):
    '''
    Returns Deep-Lift attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
            Returns:
                    array
    '''
    return DeepLift(model).attribute(inp, target=target_class)

def get_inp_gr(model, inp, target_class):
    '''
    Returns Input-Gradient attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
            Returns:
                    array
    '''
    return InputXGradient(model).attribute(inp, target=target_class)

def get_gbp(model, inp, target_class):
    '''
    Returns Guided-Backprop attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
            Returns:
                    array
    '''
    return GuidedBackprop(model).attribute(inp, target=target_class)

def get_ggc(model, inp, target_class):
    '''
    Returns Guided-GradCam attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
            Returns:
                    array
    '''
    return GuidedGradCam(model, model.features[0]).attribute(inp, target=target_class)

def get_deconv(model, inp, target_class):
    '''
    Returns Deconvolution attribution map
            Parameters:
                    model (PyTorch model): trained classifier
                    inp (PyTorch Tensor): input image as 4D PyTorch tensor
                    target_class (int): target class for which we need attribution map
            Returns:
                    array
    '''
    return Deconvolution(model).attribute(inp, target=target_class)

def get_predic_and_maps(model, inp, device_I, I_ind, device):
    '''
    Returns predicted class, list of attribution maps to ensemble
            Parameters:
                    model (PyTorch model): trained classifier
                    inp : input image
                    device_I (str): device('cpu' or 'cuda') on which attribution maps will be computed
                    I_ind (list): list of names of attribution maps     
                    device (str): device('cpu' or 'cuda') for model inference   
            Returns:
                    target_class (int): predicted_class
                    I_ALL (array): list of attribution maps
    '''
    def get_proc_map(a):
        '''
        Returns 3d attribution map converted to 2d via standard procedure of summing channels after taking abs.
                Parameters:
                        a (array): 3D attribution map
                Returns:
                        I (array): processed 2D attribution map
        '''
        I = np.sum(np.abs(np.transpose(a.squeeze(0).detach().cpu().numpy(), (1,2,0))), axis=2)
        I = (I-I.min())/(I.max()-I.min())
        return I
    model.eval()
    model = model.to(device)
    inp = inp.to(device)

    model_inp = model(inp)

    output = F.softmax(model_inp, dim = 1)
    #print(output.cpu().detach().numpy()[0])
    top_5 = np.argpartition(output.cpu().detach().numpy()[0], -5)[-5:]
    #print(top_5)
    #print(output.cpu().detach().numpy()[0][top_5])

    target_class = torch.argmax(output).item()
    input_score = model_inp.cpu().detach().numpy()[0][target_class]
    #f0 = (F.softmax(model(torch.zeros_like(inp)), dim = 1)).max().item()

    inp.requires_grad = True 
    model = model.to(device_I)
    inp = inp.to(device_I)
    
    I_ALL = []
    if "ig" in I_ind:
        ig = get_ig(model, inp, target_class)
        I_ALL.append(ig)
    if "dl" in I_ind:
        dl = get_dl(model, inp, target_class)
        I_ALL.append(dl)
    if "inp_gr" in I_ind:
        inp_gr = get_inp_gr(model, inp, target_class)
        I_ALL.append(inp_gr)
    if "ggc" in I_ind:
        ggc = get_ggc(model, inp, target_class)
        I_ALL.append(ggc)
    if "deconv" in I_ind:
        deconv = get_deconv(model, inp, target_class)
        I_ALL.append(deconv)
    if "gbp" in I_ind:
        gbp = get_gbp(model, inp, target_class)
        I_ALL.append(gbp)
    if "vg" in I_ind:
        vg = get_vg(model, inp, target_class)
        I_ALL.append(vg)
    if "sg" in I_ind:
        model = model.to("cpu")
        inp = inp.to("cpu")
        sg = get_sg(model, inp, target_class, 10)
        model = model.to(device)
        inp = inp.to(device)
        I_ALL.append(sg)
    if "sg3" in I_ind:
        sg = get_sg(model, inp, target_class, 3)
        I_ALL.append(sg)       
    if "svg" in I_ind:
        model = model.to("cpu")
        inp = inp.to("cpu")
        sg = get_svg(model, inp, target_class, 10)
        model = model.to(device)
        inp = inp.to(device)
        I_ALL.append(sg)        
    
    I_ALL = [get_proc_map(m) for m in I_ALL]
    
    model = model.to(device)
    inp = inp.to(device)
#     inp.requires_grad = False
    return target_class, I_ALL, top_5, input_score #f0, I_ALL

def get_sub_to_orig(sp_w, sp_h, sub):
    '''
    Returns original sized attribution map by upsampling the sub-sampled one
            Parameters:
                    sp_w (int): super-pixel width
                    sp_h (int): super-pixel height
                    sub (array): sub-sampled attribution map
            Returns:
                    array
    '''
    return sub.repeat(sp_h, axis=0).repeat(sp_w, axis=1)

def get_hard_thr_attr(attr, th):
    '''
    Returns hard-thresholded(binary) map
            Parameters:
                    attr (array): attribution map
                    th (int): threshold on no. of top pixels wanted
            Returns:
                    ht (array): hard-thresholded map with 1 at positions which had attribution values in top-th
    '''
    ht = np.zeros(attr.shape[0]*attr.shape[1])
    ht[attr.reshape(-1).argsort()[-th:]] = 1
    return ht.reshape(attr.shape[0], attr.shape[1])

def vanilla_final_ht_proc(thresholds, I_ALL):
    ht_all = {}
    for th in thresholds:
        ht_all[th] = []
        for i in range(0, len(I_ALL)):
            ht_all[th].append(get_hard_thr_attr(I_ALL[i], th))
    return ht_all

def final_ht_proc(sp_w, sp_h, thresholds, I_ALL):
    '''
    Returns a dictionary with keys as threshold and values as a dictionary of attribution maps sub-sampled hard-thresholded attribution maps
            Parameters:
                    sp_w (int): super-pixel width
                    sp_h (int): super-pixel height
                    thresholds (list): list of thresholds for hard-thresholding
                    I_ALL (list): list of attribution maps
            Returns:
                    ht_all (dictionary)
    '''
    def get_sub_bin(sp_w, sp_h, ht_attr):
        '''
        Returns sub-sampled hard-thresholded attribution map.
                Parameters:
                        sp_w (int): super_pixel width
                        sp_h (int): super_pixel height
                        ht_attr (array): hard-thresholded attribution map
                Returns:
                        sub_ht_map (array): sub-sampled hard-thresholded attribution map
        '''
        tot_1 = ht_attr.sum()
        new_w = int(ht_attr.shape[0]/sp_w)
        new_h = int(ht_attr.shape[1]/sp_h)
        sub_ht_map = np.zeros((new_w, new_h))
        '''
        TO-DO: remove loops
        '''
        for i in range(new_w):
            for j in range(new_h):
                sbin = ht_attr[(i*sp_h):(i*sp_h+sp_w), (j*sp_w):(j*sp_w+sp_h)]
                bin_tot_1 = sbin.sum()
                sub_ht_map[i, j] = int(bin_tot_1*new_w*new_h>=tot_1)
        return sub_ht_map
        
    ht_all = {}
    for th in thresholds:
        ht_all[th] = []
        for i in range(0, len(I_ALL)):
            ht_all[th].append(get_sub_bin(sp_w, sp_h, get_hard_thr_attr(I_ALL[i], th)))
    return ht_all

def final_subI_proc(sp_w, sp_h, I_ALL):
    '''
    Returns sub-sampled attribution map
            Parameters:
                    sp_w (int): super-pixel width
                    sp_h (int): super-pixel height
                    I_ALL (list): list of attribution maps
            Returns:
                    sub_I (list): list of sub-sampled attribution maps
    '''
    def get_sub_I(sp_w, sp_h, I):
        '''
        Returns sub-sampled attribution map.
                Parameters:
                        sp_w (int): super_pixel width
                        sp_h (int): super_pixel height
                        I (array): attribution map
                Returns:
                        sub_I (array): sub-sampled attribution map
        '''
        new_w = int(I.shape[0]/sp_w)
        new_h = int(I.shape[1]/sp_h)
        sub_I = np.empty((new_w, new_h))
        '''
        TO-DO: remove loops
        '''
        for i in range(new_w):
            for j in range(new_h):
                sub_I[i, j] = np.mean((I[(i*sp_h):(i*sp_h+sp_w), (j*sp_w):(j*sp_w+sp_h)]).reshape(-1))
        return sub_I
    
    sub_I = [get_sub_I(sp_w, sp_h, I) for I in I_ALL]
    sub_I = [(I-I.min())/(I.max()-I.min()) for I in sub_I]
    return sub_I
