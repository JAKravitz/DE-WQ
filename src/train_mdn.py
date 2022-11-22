"""
NN training and validation/test
Based on MDN code: https://github.com/hardmaru/pytorch_notebooks
Laurel Hopkins Manella
June 30, 2022
"""

import torch
import numpy as np
from src.mdn_model import mdn_loss_fn, gumbel_sample
import os
import timeit


def train_model(model, x, y, optimizer, cuda):
    if cuda:
        x = x.cuda()
        y = y.cuda()

    # zero the parameter gradients
    optimizer.zero_grad()
    # forward
    alpha, sigma, mu = model(x)
    loss = mdn_loss_fn(alpha, sigma, mu, y)
    # backward
    loss.backward()

    # optimize
    optimizer.step()

    return loss


def val_model(model, x, y, cuda):
    with torch.no_grad():
        model.eval()
    if cuda:
        x = x.cuda()
        y = y.cuda()

    # forward
    alpha, sigma, mu = model(x)
    loss = mdn_loss_fn(alpha, sigma, mu, y)

    return loss


def predict(model, x, cuda, mean_mixture):
    n_samples = x.shape[0]
    with torch.no_grad():
        model.eval()
        if cuda:
            x = x.cuda()
        alpha, sigma, mu = model(x)  # ordering: pi, sigma, mu

    alpha, sigma, mu = alpha.numpy(), sigma.numpy(), mu.numpy()

    if mean_mixture:
        # weight mean estimate by all Gaussians in mixture
        n_gaus = alpha.shape[1]
        rn = np.random.randn(n_samples, n_gaus)
        y_preds = rn * sigma + mu
        y_pred = np.sum(alpha * y_preds, axis=1)
        al_unc = np.sum(alpha * sigma, axis=1)
        ep_unc = np.sum(alpha * mu**2, axis=1) - (np.sum(alpha * mu, axis=1))**2


    else:
        # take mean estimate from Gaussian with highest mass (mixing coeff.)
        k = gumbel_sample(alpha)
        indices = (np.arange(n_samples), k)
        rn = np.random.randn(n_samples)
        y_pred = rn * sigma[indices] + mu[indices]
        al_unc = sigma[indices]  # np.sum(alpha*sigma, axis=1)  # aleatoric uncertainty
        ep_unc = np.square(mu[indices] - np.sum(alpha * mu, axis=1))  # epistemic uncertainty

    return y_pred, al_unc, ep_unc


def train_mdn(model, num_epochs, x_tensor, y_tensor, x_val_tensor, y_val_tensor, optimizer,
          lr_scheduler, cuda, writer, weights_dir, model_name):
    """
    Training loop for Mixture Density Network model
    """
    for epoch in range(1, num_epochs + 1):
        loss = train_model(model, x_tensor, y_tensor, optimizer, cuda)
        val_loss = val_model(model, x_val_tensor, y_val_tensor, cuda)
        if writer is not None:
            writer.add_scalar('loss/train', loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)

        # update lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch == num_epochs:# if you want to save model every 200 epochs:  % 200 == 0:
            # save model parameters
            save_name = os.path.join(weights_dir, model_name + '_' + str(epoch) + '.ckpt')
            torch.save(model.state_dict(), save_name)
        if epoch % 100 == 0:
            # print train & val loss
            print(f'Epoch {epoch}/{num_epochs} --- train loss: {loss:.3f}, val loss: {val_loss:.3f}')

    return loss


def predict_mdn(model, x_test_tensor, y_test, dict_key, output_dict, cuda, mean_mixture):
    tic = timeit.default_timer()
    y_pred, al_uncertainty, ep_uncertainty = predict(model, x_test_tensor, cuda, mean_mixture)
    toc = timeit.default_timer()
    print(f"Inference time (sec) for {len(y_test)} examples: {toc-tic}\n")
    output_dict[dict_key+'_al'] = {'y_pred': y_pred, 'y_test': y_test, 'uncertainty': al_uncertainty}
    output_dict[dict_key+'_ep'] = {'y_pred': y_pred, 'y_test': y_test, 'uncertainty': ep_uncertainty}
    output_dict[dict_key] = {'y_pred': y_pred, 'y_test': y_test,
                            'uncertainty': np.sqrt(al_uncertainty + ep_uncertainty)}
    return output_dict


# commenting out for now, can use batch dataloader if dataset size becomes an issue
"""
# same code as above but with a dataloader
    def train_model_dataloader(model, train_dataloader, optimizer, cuda):
    n_batches = len(train_dataloader)
    running_loss = 0.0

    for data in train_dataloader:
        inputs, labels = data
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        alpha, sigma, mu = model(inputs)
        loss = mdn_loss_fn(alpha, sigma, mu, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        running_loss += loss.item()

    return running_loss / n_batches


def val_model_dataloader(model, val_dataloader, cuda):
    n_batches = len(val_dataloader)
    running_loss = 0.0

    with torch.no_grad():
        model.eval()
        for data in val_dataloader:
            inputs, labels = data
            if cuda:
                inputs = inputs.cuda()
            alpha, sigma, mu = model(inputs)
            loss = mdn_loss_fn(alpha, sigma, mu, labels)
            running_loss += loss.item()

    ave_loss = running_loss / n_batches
    return ave_loss


def predict_dataloader(model, val_dataloader, cuda):
    n_samples = len(val_dataloader.dataset)
    y_true, mdn_outputs = [], []
    with torch.no_grad():
        model.eval()
        for data in val_dataloader:
            inputs, labels = data
            y_true.append(labels.detach().tolist())
            if cuda:
                inputs = inputs.cuda()
            outputs = model(inputs)  # ordering: pi, sigma, mu
            mdn_outputs.append(outputs.detach().numpy())

    # find Gaussian with maximum mass for each example
    alpha = mdn_outputs[:, 0]  # TODO: validate this
    sigma = mdn_outputs[:, 1]
    mu = mdn_outputs[:, 2]
    k = gumbel_sample(alpha)

    indices = (np.arange(n_samples), k)
    rn = np.random.randn(n_samples)
    y_pred = rn * sigma[indices] + mu[indices]
    return y_true, y_pred
"""