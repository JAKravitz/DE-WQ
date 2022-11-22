"""
Mixture density network (MDN) model for water quality retrieval
Code based on from: https://github.com/hardmaru/pytorch_notebooks and
                    https://github.com/tonyduan/mdn

Laurel Hopkins Manella
July 5, 2022
"""

import numpy as np
import torch
import torch.nn as nn


class MDN(nn.Module):
    def __init__(self, n_in, n_hidden, n_gaussians, n_lin_layers=4):
        super(MDN, self).__init__()
        #self.z_h = nn.Sequential()  # Use commented out code over modules if modules are problematic
        modules = []
        for i in range(0, n_lin_layers):
            if i == 0:
                #self.z_h.append(nn.Linear(n_in, n_hidden))
                modules.append(nn.Linear(n_in, n_hidden))
            else:
                #self.z_h.append(nn.Linear(n_hidden, n_hidden))
                modules.append(nn.Linear(n_hidden, n_hidden))
            #self.z_h.append(nn.Tanh())
            modules.append(nn.Tanh())
        self.z_h = nn.Sequential(*modules)
        self.z_alpha = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        alpha = nn.functional.softmax(self.z_alpha(z_h), -1)
        sigma = self.z_sigma(z_h)
        sigma = nn.ELU()(sigma) + 1 + 1e-15  # helps with numerical stability
        #sigma = torch.exp(sigma)  # alternative to ELU
        mu = self.z_mu(z_h)
        return alpha, sigma, mu


oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi)  # normalization factor for Gaussians
def gaussian_distribution(y, mu, sigma):
    if len(y.shape) == 1:  # single target variable
        y = torch.unsqueeze(y, 1)  # add a second dim
    y = y.expand_as(sigma)  # TODO: fix for >1 target variable

    ret = (oneDivSqrtTwoPI / sigma) * torch.exp(
        -0.5 * ((y - mu) / sigma) ** 2)
    return ret


def mdn_loss_fn(alpha, sigma, mu, y):
    prob = alpha * gaussian_distribution(y, mu, sigma)
    nll = -torch.log(torch.sum(prob, dim=1))
    nll_mean = torch.mean(nll)
    return nll_mean


def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)