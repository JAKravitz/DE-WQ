"""
Deep Ensemble Model
Laurel Hopkins Manella
July 29, 2022
"""

import torch
import os
import numpy as np
from src.de_model import NLLloss
import timeit


def train_model(model, x, y, optimizer, cuda):
    if cuda:
        x = x.cuda()
        y = y.cuda()

    # zero the parameter gradients
    optimizer.zero_grad()
    # forward
    mu, sigma = model(x)
    loss = NLLloss(y, mu, sigma)

    # backward
    loss.backward()
    # optimize
    optimizer.step()

    return loss.item()


def val_model(model, x, y, cuda):
    with torch.no_grad():
        model.eval()
    if cuda:
        x = x.cuda()
        y = y.cuda()

    # forward
    mu, sigma = model(x)
    loss = NLLloss(y, mu, sigma)

    return loss.item()


def predict(gmm, x, cuda):
    mus, sigma_sqs = [], []
    for i in range(gmm.num_models):
        model = getattr(gmm, 'model_' + str(i))

        with torch.no_grad():
            model.eval()
            if cuda:
                x = x.cuda()

        mu, sigma_sq = model(x)
        mu = mu.detach().numpy()
        sigma_sq = sigma_sq.detach().numpy()
        mus.append(mu)
        sigma_sqs.append(sigma_sq)

    mus = np.squeeze(np.array(mus))
    sigma_sqs = np.squeeze(np.array(sigma_sqs))
    mixture_mus = np.mean(mus, axis=0)
    mixture_sigmas = np.mean((sigma_sqs + mus**2), axis=0) - mixture_mus**2
    mixture_sigmas = np.sqrt(mixture_sigmas)

    return mixture_mus, mixture_sigmas


def train_de(gmm_model, num_epochs, x_tensor, y_tensor, x_val_tensor, y_val_tensor, gmm_optimizers,
              lr_schedulers, cuda, writer, weights_dir, model_name):
    """
    Training loop for Deep Ensemble model
    """
    for epoch in range(1, num_epochs + 1):
        losses, val_losses = [], []
        # iterate over each model in ensemble
        for i in range(gmm_model.num_models):
            model = getattr(gmm_model, 'model_' + str(i))
            loss = train_model(model, x_tensor, y_tensor, gmm_optimizers[i], cuda)
            losses.append(loss)
            val_loss = val_model(model, x_val_tensor, y_val_tensor, cuda)
            val_losses.append(val_loss)
            if writer is not None:
                writer.add_scalar(f'loss/val_{i}', val_loss, epoch)
                writer.add_scalar(f'loss/train_{i}', loss, epoch)

        # update lr scheduler
        if lr_schedulers is not None:
            lr_schedulers[i].step()

        if epoch == num_epochs:# if you want to save model every 200 epochs:  % 200 == 0:
            # save model parameters
            for i in range(gmm_model.num_models):
                save_name = os.path.join(weights_dir, model_name + '_' + str(i) + '_' + str(epoch) + '.ckpt')
                model = getattr(gmm_model, 'model_' + str(i))
                torch.save(model.state_dict(), save_name)
        if epoch % 100 == 0:
            # print train & val loss
            losses = [round(x, 3) for x in losses]
            val_losses = [round(x, 3) for x in val_losses]
            print(f'Epoch {epoch}/{num_epochs} --- train loss: {(*losses,)}, val loss: {(*val_losses,)}')

    return loss


def predict_de(model, x_test_tensor, y_test, dict_key, output_dict, cuda):
    # predict on test set
    tic = timeit.default_timer()
    mus, sigmas = predict(model, x_test_tensor, cuda)
    toc = timeit.default_timer()
    print(f"Inference time (sec) for {len(y_test)} examples: {toc-tic}\n")
    output_dict[dict_key] = {'y_pred': mus, 'y_test': y_test, 'uncertainty': sigmas}

    return output_dict
