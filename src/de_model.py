"""
Deep Ensemble (DE) model for water quality retrieval
Code based on from: https://github.com/mpritzkoleit/deep-ensembles

Laurel Hopkins Manella
July 28, 2022
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMLP(nn.Module):
    """ Gaussian MLP which outputs are mean and variance.

    Attributes:
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(self, inputs, outputs, hidden_layers, n_lin_layers):
        super(GaussianMLP, self).__init__()
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.outputs = 2*outputs
        modules = []
        for i in range(0, n_lin_layers):
            if i == 0:
                modules.append(nn.Linear(self.inputs, self.hidden_layers))
            elif i == n_lin_layers - 1:
                modules.append(nn.Linear(self.hidden_layers, self.outputs))
            else:
                modules.append(nn.Linear(self.hidden_layers, self.hidden_layers))
            if i < n_lin_layers - 1:
                modules.append(nn.ReLU())

        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        x = self.seq(x)
        mu, sigma = torch.tensor_split(x, self.outputs, dim=1)
        sigma = F.softplus(sigma) + 1e-6
        return mu, sigma


class DE(nn.Module):
    """ Deep Ensemble: Gaussian mixture MLP which outputs are mean and variance.

    Attributes:
        models (int): number of models
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(self, num_models=5, inputs=1, outputs=1, hidden_layers=[100], n_lin_layers=2):
        super(DE, self).__init__()
        self.num_models = num_models
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.n_lin_layers = n_lin_layers
        for i in range(self.num_models):
            model = GaussianMLP(inputs=self.inputs,
                                outputs=self.outputs,
                                hidden_layers=self.hidden_layers,
                                n_lin_layers=self.n_lin_layers)
            setattr(self, 'model_' + str(i), model)


def NLLloss(y, mu, sigma):
    """ Negative log-likelihood loss function. """
    return torch.mean(0.5*torch.log(sigma) + 0.5 * ((y - mu).pow(2)/sigma))
