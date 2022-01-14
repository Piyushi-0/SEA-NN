import torch, copy
import numpy as np
from utils import pre_process

def c_sb_mx(f, Klist, sq_n_sb_px, device):
    '''
    Returns solution of cardinality constrained submodular maximization
            Parameters:
                    f (PyTorch model): DSF
                    Klist (list): List of cardinalities
                    sq_n_sb_px (int): one-side dimension of the map
                    device (str): device('cpu' or 'cuda') on which to run
            Returns:
                    AList (dic): Dictionary with keys as cardinalities and values as solutions  
    '''
    k = int(np.array(Klist).max())#we only need to solve for max cardinality
    card_V = sq_n_sb_px*sq_n_sb_px#cardinality of V
    x = torch.eye(card_V)#card_V number of candidate A's each arranged in columns
    mx = f(torch.zeros(sq_n_sb_px, sq_n_sb_px).view(1, 1, sq_n_sb_px, sq_n_sb_px).double().to(device)).item()#f(A), initially A is {}
    AList = {}#dic with key k, value A*_k where A*_k is the optimal subset of cardinality k
    '''
    Note:
    - inputs contain all possible candidate A's
    - v: index of chosen pixel argmax_{v \in (V\A)} f({v} U A)
    - {v} U A is maintained in x
    - initially A is {}
    '''
    for it in range(1, k+1):
        inputs = x.t().view(card_V, 1, sq_n_sb_px, sq_n_sb_px)#'x' reshaped as PyTorch input
        outputs = f(torch.Tensor(inputs).double().to(device))
        v = outputs.argmax(dim=0).item()
        if outputs[v]>mx:
            mx = outputs[v]
            selected = x[:, v]
            x[v, :] = 1
            if it in Klist:
#                 AList[it] = selected
                AList[it] = selected.detach().clone()
        else:
            break
    try:
        for it in Klist:
            if it not in AList:
                print("Putting max till now for {}".format(it))
#                 AList[it] = selected
                AList[it] = selected.detach().clone()    
        return AList
    except:
        print("EmptySet{}".format(outputs[v].item()))
        return torch.zeros(sq_n_sb_px*sq_n_sb_px)

def sea_nn(f, sq_n_sb_px, sp_w, sp_h, device):
    '''
    Returns Submodular Ensembled Attribution
            Parameters:
                    f (PyTorch model): DSF
                    sq_n_sb_px (int): one-side dimension of the map
                    device (str): device('cpu' or 'cuda') on which to run
            Returns:
                    orig_scale_G (array): Submodular Ensembled Attribution map
    '''
    card_V = sq_n_sb_px*sq_n_sb_px
    x = torch.eye(card_V)
    mx = f(torch.zeros(sq_n_sb_px, sq_n_sb_px).view(1, 1, sq_n_sb_px, sq_n_sb_px).double().to(device)).item()
    AList = {}
    G = np.zeros(card_V)
    for it in range(card_V):
        inputs = x.t().view(card_V, 1, sq_n_sb_px, sq_n_sb_px)
        outputs = f(torch.Tensor(inputs).double().to(device)).cpu().detach().numpy()
        v = outputs.argmax()
        if outputs[v]>mx:
            gain = outputs[v]-mx
            mx = outputs[v]
            all_max = [copy.deepcopy(i[0]) for i in np.argwhere(outputs==mx)]
            x[v,:] = 1
            G[all_max] = gain
        else:
            break
    orig_scale_G = pre_process.get_sub_to_orig(sp_w, sp_h, G.reshape(sq_n_sb_px, sq_n_sb_px))
    orig_scale_G = (orig_scale_G-orig_scale_G.min())/(orig_scale_G.max()-orig_scale_G.min())
    return orig_scale_G
