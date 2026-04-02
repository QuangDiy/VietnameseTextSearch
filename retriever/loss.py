from enum import Enum
import torch

def cosine_similarity(x, y):
    return torch.mm(x/x.norm(dim=-1)[:,None], (y/y.norm(dim=-1)[:,None]).transpose(0,1))

def gteloss(x, y, z, temperature=0.1):
    a_mat = torch.exp(cosine_similarity(x, y) / temperature)
    x_mat = torch.exp(cosine_similarity(x, x) / temperature)
    y_mat = torch.exp(cosine_similarity(y, y) / temperature)
    loss = -torch.log(a_mat.diagonal() / (a_mat.sum(dim=-1) + a_mat.sum(dim=0) + x_mat.sum(dim=0) + y_mat.sum(dim=0))).mean() # - x_mat.diagonal() - y_mat.diagonal()
    return loss

def InfoNCEloss(x, y, z, temperature=0.1):
    exp_sim = torch.exp(cosine_similarity(x, y) / temperature)
    return -torch.log(exp_sim.diagonal() / exp_sim.sum(dim=-1)).mean()

def monoeloss(x, y, z, strength=torch.e):
    sim_mat = torch.exp(cosine_similarity(x, y))
    batch_size = x.shape[0]

    win = sim_mat.diag()
    lose = torch.diagonal_scatter(sim_mat, torch.full((batch_size,), -torch.inf), 0)

    return torch.exp(strength-win).mean() + torch.exp(lose).mean()

def eloss(x, y, z, strength=torch.e):
    return (monoeloss(x, y, strength) + monoeloss(x, x, strength) + monoeloss(y, y, strength)) / 3

class Losses(Enum):
    ELOSS = eloss
    GTELOSS = gteloss
    MONOELOSS = monoeloss
    INFONCELOSS = InfoNCEloss