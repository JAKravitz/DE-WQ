import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

home_dir = r'C:\Users\laurelm\Documents\WaterQuality'

model = 'mdn'  # options: mdn, de, tutorial_mdn (hardcoded mdn from tutorial)
num_epochs = 10000

model_name = 'toy_' + model

if model == 'mdn':
    from src.mdn_model import MDN
    from src.train_mdn import train_mdn, predict
elif model == 'de':
    from src.de_model import DE
    from src.train_de import train_de, predict

# create directory for saved models
saved_model_dir = os.path.join(home_dir, model_name)
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
# create directory for model weights
model_weights_dir = os.path.join(saved_model_dir, 'model_weights')
if not os.path.exists(model_weights_dir):
    os.makedirs(model_weights_dir)


def generate_data(n_samples):
    epsilon = np.random.normal(size=(n_samples))
    x_data = np.random.uniform(-10.5, 10.5, n_samples)
    y_data = 7 * np.sin(0.75 * x_data) + 0.5 * x_data + epsilon
    return x_data, y_data


n_samples = 1000
x_data, y_data = generate_data(n_samples)

# plot generated data
plt.figure(figsize=(8, 8))
plt.scatter(x_data, y_data, alpha=0.2)
plt.show()

# define NN model
n_input = 1
n_hidden = 20
n_output = 1

loss_fn = nn.MSELoss()

# 1. Train basic NN
network = nn.Sequential(nn.Linear(n_input, n_hidden),
                        nn.Tanh(),
                        nn.Linear(n_hidden, n_output))

optimizer = torch.optim.RMSprop(network.parameters())

# change data type and shape, move from numpy to torch
# note that we need to convert all data to np.float32 for pytorch
x_tensor = torch.from_numpy(np.float32(x_data).reshape(n_samples, n_input))
y_tensor = torch.from_numpy(np.float32(y_data).reshape(n_samples, n_input))
x_variable = Variable(x_tensor)
y_variable = Variable(y_tensor, requires_grad=False)

def train():
    for epoch in range(3000):
        y_pred = network(x_variable) # make a prediction
        loss = loss_fn(y_pred, y_variable) # compute the loss
        optimizer.zero_grad() # prepare the optimizer
        loss.backward() # compute the contribution of each parameter to the loss
        optimizer.step() # modify the parameters

        if epoch % 300 == 0:
            print(epoch, loss.item())

train()

# evenly spaced samples from -10 to 10 (test data)
x_test_data = np.linspace(-10, 10, n_samples)

# change data shape, move from numpy to torch
x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
x_test_variable = Variable(x_test_tensor)
y_test_variable = network(x_test_variable)

# move from torch back to numpy
y_test_data = y_test_variable.data.numpy()

# plot the original data and the test data
plt.figure(figsize=(8, 8))
plt.scatter(x_data, y_data, alpha=0.2)
plt.scatter(x_test_data, y_test_data, alpha=0.2)
plt.show()


# 2. Rotate sinusoid 90 degrees
# plot x against y instead of y against x
plt.figure(figsize=(8, 8))
plt.scatter(y_data, x_data, alpha=0.2)
plt.show()

x_variable.data = y_tensor
y_variable.data = x_tensor

train()

x_test_data = np.linspace(-15, 15, n_samples)
x_test_tensor = torch.from_numpy(np.float32(x_test_data).reshape(n_samples, n_input))
x_test_variable.data = x_test_tensor

y_test_variable = network(x_test_variable)

# move from torch back to numpy
y_test_data = y_test_variable.data.numpy()

# plot the original data and the test data
plt.figure(figsize=(8, 8))
plt.scatter(y_data, x_data, alpha=0.2)
plt.scatter(x_test_data, y_test_data, alpha=0.2)
plt.show()

# 3. MDN
# tutorial MDN
class tutorial_MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(tutorial_MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu

oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians
def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI

def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


# define model
if model == 'mdn':
    network = MDN(n_in=1, n_hidden=20, n_gaussians=5, n_lin_layers=1)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
elif model == 'de':
    network = DE(num_models=10, inputs=1, hidden_layers=20, n_lin_layers=2)
    learning_rate = 0.001
    optimizer = []
    for i in range(network.num_models):
        ensemble_mem = getattr(network, 'model_' + str(i))
        opt = torch.optim.Adam(params=ensemble_mem.parameters(), lr=learning_rate)
        optimizer.append(opt)
elif model == 'tutorial_mdn':
    network = tutorial_MDN(n_hidden=20, n_gaussians=5)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

print('\n', network)

mdn_x_data = y_data
mdn_y_data = x_data

mdn_x_tensor = y_tensor
mdn_y_tensor = x_tensor

x_variable = Variable(mdn_x_tensor)
y_variable = Variable(mdn_y_tensor, requires_grad=False)


def train_tutorial_mdn():
    for epoch in range(num_epochs):
        pi_variable, sigma_variable, mu_variable = network(x_variable)
        loss = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, y_variable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(epoch, loss.item())

# train
if model == 'mdn':
    train_mdn(network, num_epochs, x_variable, y_variable, x_variable, y_variable, optimizer,
              None, False, None, model_weights_dir, model_name)
elif model == 'de':
    train_de(network, num_epochs, x_variable, y_variable, x_variable, y_variable, optimizer,
              None, False, None, model_weights_dir, model_name)
elif model == 'tutorial_mdn':
    train_tutorial_mdn()

# predict
if model == 'mdn':
    sampled, _, _ = predict(network, x_test_variable, False, mean_mixture=False)
elif model == 'de':
    sampled, _ = predict(network, x_test_variable, False)
elif model == 'tutorial_mdn':
    pi_variable, sigma_variable, mu_variable = network(x_test_variable)

def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)

if model == 'tutorial_mdn':
    pi_data = pi_variable.data.numpy()
    sigma_data = sigma_variable.data.numpy()
    mu_data = mu_variable.data.numpy()

    k = gumbel_sample(pi_data)

    indices = (np.arange(n_samples), k)
    rn = np.random.randn(n_samples)
    sampled = rn * sigma_data[indices] + mu_data[indices]

# plot
plt.figure(figsize=(8, 8))
plt.scatter(mdn_x_data, mdn_y_data, alpha=0.2)
plt.scatter(x_test_data, sampled, alpha=0.2, color='red')
plt.show()