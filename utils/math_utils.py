import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy.sparse import issparse, csc_matrix

def softmax_entropy(x, enable_softmax=True, temperature=1.0):
    if enable_softmax:
        probs = torch.softmax(x / temperature, dim=1) 
    else:
        probs = x
    return -(probs * torch.log(probs + 1e-10)).sum(dim=1)

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)

def online_value_refinement_new(cache_keys, all_probs, add_new_center, L_reg_old, L_reg_old_inv, iteration, threshold=0.5, lambda_reg=0.13, k=2):
    device = cache_keys.device
    normalized_keys = F.normalize(cache_keys, p=2, dim=1)
    W = torch.mm(normalized_keys, normalized_keys.T)
    W[W < threshold] = 0
    
    # Normalized Laplacian
    D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(W.sum(dim=1)) + 1e-8))
    L_norm = torch.eye(W.size(0), device=device) - D_inv_sqrt @ W @ D_inv_sqrt
    I = torch.eye(L_norm.size(0), device=device)   
    L_reg = L_norm + 2 * lambda_reg * I
    L_reg = L_reg.float()

    if add_new_center == True: 
        L_inv = torch.linalg.inv(L_reg)
    elif add_new_center == False:
        # Recursive inverse update approximation
        delta_L = L_reg - L_reg_old
        
        if issparse(L_reg_old_inv) or issparse(delta_L):
            L_reg_old_inv = csc_matrix(L_reg_old_inv)
            delta_L = csc_matrix(delta_L)

        if k == 0:
            L_reg_new_inv = L_reg_old_inv 
        elif k == 1:
            L_reg_new_inv = L_reg_old_inv - L_reg_old_inv @ delta_L @ L_reg_old_inv
        elif k == 2:
            term1 = L_reg_old_inv
            term2 = L_reg_old_inv @ delta_L @ L_reg_old_inv
            term3 = L_reg_old_inv @ L_reg_old_inv @ delta_L @ L_reg_old_inv @ delta_L
            L_reg_new_inv = term1 - term2 + term3
        
        L_inv = L_reg_new_inv

    all_probs_new = L_inv @ (2 * lambda_reg * all_probs)
    all_probs_new = all_probs_new / all_probs_new.sum(1).unsqueeze(1)

    return all_probs_new, (L_reg, L_inv)

def online_value_refinement_old(cache_keys, all_probs, threshold=0.5, lambda_reg=0.13):
    device = cache_keys.device
    normalized_keys = F.normalize(cache_keys, p=2, dim=1)
    W = torch.mm(normalized_keys, normalized_keys.T)
    W[W < threshold] = 0
    
    D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(W.sum(dim=1)) + 1e-8))
    L_norm = torch.eye(W.size(0), device=device) - D_inv_sqrt @ W @ D_inv_sqrt
    I = torch.eye(L_norm.size(0), device=device)   
    L_reg = L_norm + 2 * lambda_reg * I
    L_reg = L_reg.float()
    L_inv = torch.linalg.inv(L_reg)

    all_probs_new = L_inv @ (2 * lambda_reg * all_probs)
    all_probs_new = all_probs_new/all_probs_new.sum(1).unsqueeze(1)

    return all_probs_new