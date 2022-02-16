import numpy as np
import copy
import torch.nn.functional as F
from sklearn.metrics import auc
import torch
import random
import math
from torch.utils.data import DataLoader
from scipy.integrate import simps
from skimage.transform import pyramid_gaussian
import torchvision.transforms as transforms

"""
Code to compute Area under the perturbation curve. Inspired by the code of the following paper:
Goh, S. W. Goh, S. Lapuschkin, L. Weber, W. Samek, and A. Binder (2021). “Understanding Integrated Gradients with SmoothTaylor for Deep Neural Network Attribution”. In: 2020 25th International Conference on Pattern Recognition (ICPR), pp. 4949–4956. DOI:10.1109/ICPR48806.2021.9413242.
Link to their code: https://github.com/garygsw/smooth-taylor/blob/master/attribution/eval.py
"""

DEVICE = "cuda"

NORMALIZE_TRANSFORM = transforms.Compose([
    transforms.Normalize(         # Normalize by setting the mean and s.d. to specified values
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
])

INVERSE_TRANSFORM = transforms.Compose([
    transforms.Normalize(         # Normalize by setting the mean and s.d. to specified values
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)
])

def classify_perturbations(data_loader, model, explained_class):
    all_scores = []
    for sample_batch in data_loader:
        inputs = sample_batch[0]
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            out = model(inputs)
            scores = out[:, explained_class].cpu().numpy()
            all_scores.append(scores)
    all_scores = np.array(all_scores)
    all_scores = np.concatenate(all_scores)

    mean_score = np.mean(all_scores)
    median = np.median(all_scores)
    best_idx = 0
    best_val = np.inf
    for i in range(len(all_scores)):
        value = math.fabs(all_scores[i]-median)
        if value<best_val:
            best_val = value 
            best_idx = i
    return best_idx, mean_score

def compute_aupc(heatmap, img_input, model, batch_size, explained_class, input_score, kernel_size, draw_mode, num_regions, num_perturbs, pert_upto=3):
    perturb_scores = compute_perturbations(
            img_input = img_input,
            model = model,
            batch_size = batch_size,
            explained_class = explained_class,
            heatmap = heatmap,
            kernel_size = kernel_size,
            draw_mode = draw_mode,
            num_regions = num_regions,
            num_perturbs = num_perturbs
        )

    perturb_scores = [input_score]+perturb_scores
    perturb_scores = [x/math.fabs(input_score) for x in perturb_scores]
    aupc = simps(perturb_scores[:pert_upto], dx = 1)
    return aupc, perturb_scores

def compute_perturbations(img_input, model, batch_size, explained_class, heatmap,
                          kernel_size, draw_mode, num_regions, num_perturbs, FMNIST = False):
    img_h, img_w = heatmap.shape
    avg_values = -np.inf*np.ones(img_h*img_w)
    for h in range(img_h-kernel_size):
        for w in range(img_w-kernel_size):
            avg_values[h+w*img_h] = np.mean(np.abs(heatmap[h:h+kernel_size, w:w+kernel_size]))
    most_relevant_idxs = np.argsort(-avg_values)

    if len(img_input.shape)==4:
        img_input = img_input.squeeze(0)

    img = INVERSE_TRANSFORM(img_input)
    if draw_mode == 1:
        if FMNIST:
            channel_stats = np.array([np.mean(img), np.std(img)])
            if np.isnan(channel_stats[1]):
                channel_stats[1] = 1e-3
        else:
            channel_stats = np.zeros((2, 3))
            for c in range(3):
                channel_stats[0, c] = np.mean(img[c, :, :])
                channel_stats[1, c] = np.std(img[c, :, :])
                if np.isnan(channel_stats[1, c]):
                    channel_stats[1, c] = 1e-3
    perturb_scores = []
    
    bad_idxs = set()
    for region_idx in range(num_regions):
        found = False
        for i, kernel_idx in enumerate(most_relevant_idxs):
            if kernel_idx not in bad_idxs:
                width = int(math.floor(kernel_idx/img_h))
                height = int(kernel_idx-width*img_h)

                if (img_h-height)<=kernel_size or (img_w-width)<=kernel_size:
                    continue

                for h in range(-kernel_size+1, kernel_size):
                    for w in range(-kernel_size+1, kernel_size):
                        bad_idxs.add((height+h)+(width+w)*img_h)
                found = True
                break
        if not found:
            break

        perturbs = []
        perturbs_imgs = torch.stack([torch.zeros_like(img_input) for _ in range(num_perturbs)])

        for i in range(num_perturbs):
            if draw_mode == 0:
                if FMNIST:
                    perturb = np.random.uniform(low = 0, high = 255, size = (kernel_size, kernel_size))
                else:
                    perturb = np.random.uniform(low = 0, high = 255, size = (3, kernel_size, kernel_size))
            elif draw_mode == 1:
                if FMNIST:
                    #perturb = np.zeros((kernel_size, kernel_size))
                    perturb = np.random.normal(loc = channel_stats[0], scale = channel_stats[1], size = (kernel_size, kernel_size))
                else:
                    perturb = np.zeros((3, kernel_size, kernel_size))
                    for c in range(3):
                        perturb[c] = np.random.normal(loc = channel_stats[0, c], scale = channel_stats[1, c], size = (kernel_size, kernel_size))
                perturb = np.maximum(perturb, np.zeros_like(perturb))
                perturb = np.minimum(perturb, 255*np.ones_like(perturb))
            elif draw_mode == 2:
                assert not FMNIST

                perturb = np.zeros((3, kernel_size, kernel_size))
                for c in range(3):
                    perturb[c] = np.mean(img[c, :, :].cpu().detach().numpy())

                perturb = np.maximum(perturb, np.zeros_like(perturb))
                perturb = np.minimum(perturb, 255*np.ones_like(perturb))
            else:
                print('Invalid perturb draw mode')
                exit()

            perturbs_imgs[i] = img
            perturb = perturb/255.
            if FMNIST:
                perturbs_imgs[i, :, height:height+kernel_size, width:width+kernel_size] = torch.Tensor(perturb)
            else:
                perturbs_imgs[i, :, height:height+kernel_size, width:width+kernel_size] = torch.Tensor(perturb)
            perturbs_imgs[i] = NORMALIZE_TRANSFORM(perturbs_imgs[i])
            perturbs.append(perturb)

        perturb_dataset = torch.utils.data.dataset.TensorDataset(perturbs_imgs)
        data_loader = DataLoader(perturb_dataset, batch_size = batch_size, shuffle = False)
        actual_perturb_idx, mean_score = classify_perturbations(data_loader, model, explained_class)

        actual_perturb = perturbs[actual_perturb_idx]
        if FMNIST:
            img[height:height+kernel_size, width:width+kernel_size] = torch.Tensor(actual_perturb)
        else:
            img[:, height:height+kernel_size, width:width+kernel_size] = torch.Tensor(actual_perturb)
        perturb_scores.append(mean_score)
    return perturb_scores

