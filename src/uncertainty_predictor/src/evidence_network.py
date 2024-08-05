from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

class EvidenceNetwork(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            LinearNormalGamma(64, y_dim))
    
    def forward(self,x):
        y_pred = torch.cat(self.model(x), dim=-1)
        # mu, v, alpha, beta = np.split(y_pred, 4, axis=-1)
        # dict = {'mu': mu,'v':v,'alpha':alpha,'beta':beta}
        return y_pred


class LinearNormalGamma(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_chanels, out_channels*4)

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)

    def forward(self, x):
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, self.evidence(logv), self.evidence(logalpha) + 1, self.evidence(logbeta)


def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(two_blambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)

    return nll


def nig_reg(y, gamma, v, alpha, beta):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi


def evidential_regresssion_loss(y, pred, coeff=1.0):
    gamma, v, alpha, beta = np.split(pred, 4, axis=-1) #SH_TODO: gamma == mu ? 
    loss_nll = nig_nll(y, gamma, v, alpha, beta)
    loss_reg = nig_reg(y, gamma, v, alpha, beta)
    return loss_nll.mean() + coeff * loss_reg.mean()

