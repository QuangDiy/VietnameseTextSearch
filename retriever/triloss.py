from enum import Enum
import torch
from torch.nn.functional import cosine_similarity

def full_cosine_similarity(x, y):
    return torch.mm(x/x.norm(dim=-1)[:,None], (y/y.norm(dim=-1)[:,None]).transpose(0,1))

def unitdropinf(x, y, temperature=0.4):
    xy_mat = torch.exp(full_cosine_similarity(x, y) / temperature)
    prob = xy_mat.diagonal() / xy_mat.sum(dim=-1)
    return (-torch.log(prob) * (1-prob)).mean()

def dropinfonceloss(x, y, z, temperature=0.4):
    if z is not None:
        xy_mat = torch.exp(cosine_similarity(x, y) / temperature)
        xz_mat = torch.exp(cosine_similarity(x, z) / temperature)
        
        prob = xy_mat / (xy_mat + xz_mat)

        return (-torch.log(prob) * (1-prob)).mean()
    else:
        return (unitdropinf(x, y, temperature)+unitdropinf(y, x, temperature)+unitdropinf(x, x, temperature)+unitdropinf(y, y, temperature))/4

def eloss(x, y, z):
    a_mat = torch.exp(cosine_similarity(x, y))
    x_mat = torch.exp(cosine_similarity(x, z))
    y_mat = torch.exp(cosine_similarity(y, z))

    return (torch.exp(1-a_mat)+torch.exp(1+x_mat)+torch.exp(1+y_mat)).mean() / 3.0

def ppoloss(x, y, z, temperature=1):
    xy_mat = cosine_similarity(x, y) / temperature
    xz_mat = cosine_similarity(x, z) / temperature
    return -torch.log(torch.nn.functional.sigmoid(xy_mat-xz_mat)).mean()

def infonceloss(x, y, z, temperature=0.4):
    if z is not None:
        xy_mat = torch.exp(cosine_similarity(x, y) / temperature)
        xz_mat = torch.exp(cosine_similarity(x, z) / temperature)
        return -torch.log(xy_mat / (xy_mat + xz_mat)).mean()
    else:
        xy_mat = torch.exp(full_cosine_similarity(x, y) / temperature)
        return -torch.log(xy_mat.diagonal() / xy_mat.sum(dim=-1)).mean()

def trinceloss(x, y, z, temperature=0.4):
    xy_mat = torch.exp(cosine_similarity(x, y) / temperature)
    xz_mat = torch.exp(cosine_similarity(x, z) / temperature)
    yz_mat = torch.exp(cosine_similarity(y, z) / temperature)
    return -torch.log(xy_mat / (yz_mat + xz_mat + xy_mat)).mean()

class TriLosses(Enum):
    ELOSS = eloss
    INFONCELOSS = infonceloss
    TRINCELOSS = trinceloss
    PPOLOSS = ppoloss
    DROPINFONCELOSS = dropinfonceloss