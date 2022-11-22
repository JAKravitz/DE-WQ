"""
Script for reading in SWIPE data and training model to predict several water quality targets
Based on Jeremy Kravitz's code: https://github.com/JAKravitz/JPL_WQ_ML/blob/main/TrainingData.ipynb
    and MDN code: https://github.com/hardmaru/pytorch_notebooks

Laurel Hopkins Manella
June 23, 2022
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #required for MAC M1

import warnings
warnings.filterwarnings("ignore")
from src.MLP_retrieval_PyTorch import WQ_Data
from src.mdn_model import MDN
from src.de_model import DE
from src.train_mdn import train_mdn, predict_mdn
from src.train_de import train_de, predict_de
from src.test import run_test, eval_feature_importance
from src.evaluation import *
from src.utils import *
from src.viz_utils import *
import numpy as np
import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from src.dataloader import wq_dataloader
import timeit
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
args = parser.parse_args()


# environment
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
cuda = torch.cuda.is_available()
print('Cuda available: ' + str(cuda))

# read config file
config = read_yaml(args.yaml)
batch_info = config['batch_info']

# name of model for saving parameters
model_name = config['model_name']

# create directory for saved models
saved_model_dir = os.path.join(config['home_dir'],config['model_dir'], model_name)
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
# create directory for model weights
model_weights_dir = os.path.join(saved_model_dir, 'model_weights')
if not os.path.exists(model_weights_dir):
    os.makedirs(model_weights_dir)


# retrieve data
sensorData = load_data(config['data']['input_fpath'])

# inspect datasets
print(f'\nSensors: {sensorData.keys()}\n')

targets = config['sensor_data_info']['targets']

# instantiate the data class
wqData = WQ_Data(config['sensor_data_info'])

# get the X (inputs) and Y (outputs) data
X, y, clusters = wqData.getXY(sensorData, print_stats=True)

# split data into train/test/val splits
if batch_info['predefined_split']:
    split_data = wqData.load_split(batch_info['predefined_split'])
elif batch_info['split_method'] == 'random':
    split_data = wqData.random_train_test_val_split(X, y, clusters, batch_info['test_split'],
                                                    batch_info['val_split'], saved_model_dir)
elif batch_info['split_method'] == 'stratified':
    split_data = wqData.stratified_train_test_val_split(X, y, clusters, batch_info['test_split'],
                                                        batch_info['val_split'], saved_model_dir)
elif batch_info['split_method'] == 'binned':
    split_data = wqData.binned_train_test_val_split(X, y, clusters, batch_info['test_split'],
                                                        batch_info['val_split'], saved_model_dir)
elif batch_info['split_method'] == 'water_type':
    split_data = wqData.water_type_train_test_val_split(X, y, clusters, batch_info['val_split'],
                                                        batch_info['test_water_type'], saved_model_dir)
else:
    raise Exception("Data split specified in conf['batch_info']['split_method'] is unknown. Please redefine split.")

if batch_info['predefined_split']:
    print(f"Train/test/val split: loaded from {batch_info['predefined_split']}\n")
else:
    print(f"Train/test/val split: {batch_info['split_method']}\n")

X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test, clust_val = split_data

# y = log transformed and standard scaled
# y0 = raw y values
X_train, y_train, y0_train = wqData.preprocessXY(X_train, y_train, config, saved_model_dir,
                                                 split='train', plot_hist=True)
X_test, y_test, y0_test = wqData.preprocessXY(X_test, y_test, config, saved_model_dir,
                                              split='test', plot_hist=False)
X_val, y_val, y0_val = wqData.preprocessXY(X_val, y_val, config, saved_model_dir,
                                           split='val', plot_hist=False)

# prep the results dictionary for the various evaluation statistics
results = wqData.prep_results(y)

# if using a dataloader, define as:
# dataloader significantly slows down the process, not using for now
# train_dataloader = wq_dataloader(X_train, y_train, batch_size=batch_info['batch_size'],
#                                 shuffle=True, num_workers=0)

# otherwise, convert data to tensors (this will train on entire dataset at once
# rather than in batches)
x_tensor = torch.from_numpy(np.float32(X_train))
y_tensor = torch.from_numpy(np.float32(y_train))
x_test_tensor = torch.from_numpy(np.float32(X_test))
y_test = np.float32(y_test)
x_val_tensor = torch.from_numpy(np.float32(X_val))
y_val_tensor = torch.from_numpy(np.float32(y_val))


# define model
lr_scheduler = None
if batch_info['model'] == 'mdn':
    model = MDN(n_in=wqData.n_in, n_hidden=batch_info['n_hidden'], n_gaussians=batch_info['num_gaussians'],
                n_lin_layers=batch_info['num_lin_layers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=batch_info['lrate'],
                             weight_decay=batch_info['weight_decay'])
    if batch_info['lr_decay'] != 0:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=batch_info['lr_decay'])
    # summary(model, input_size=wqData.n_in)
elif batch_info['model'] == 'de':
    if batch_info['num_lin_layers'] <= 1:
        raise Exception("DE model must have >1 linear layers.")

    model = DE(num_models=batch_info['num_gaussians'], inputs=wqData.n_in,
                               hidden_layers=batch_info['n_hidden'], n_lin_layers=batch_info['num_lin_layers'])
    optimizer = []
    if batch_info['lr_decay'] != 0:
        lr_scheduler = []
    for i in range(model.num_models):
        ensemble_mem = getattr(model, 'model_' + str(i))
        opt = torch.optim.Adam(params=ensemble_mem.parameters(), lr=batch_info['lrate'],
                         weight_decay=batch_info['weight_decay'])
        optimizer.append(opt)
        if batch_info['lr_decay'] != 0:
            lr_scheduler.append(torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=batch_info['lr_decay']))
    # summary(model, input_size=wqData.n_in)
else:
    raise Exception("Model specified in conf['batch_info']['model'] is unknown. Please redefine model.")

print('\n', model)

# logging
writer = SummaryWriter(saved_model_dir)

# train model
if batch_info['model'] == 'mdn':
    tic = timeit.default_timer()
    loss = train_mdn(model, batch_info['epochs'], x_tensor, y_tensor, x_val_tensor, y_val_tensor, optimizer,
              lr_scheduler, cuda, writer, model_weights_dir, model_name)
    toc = timeit.default_timer()
elif batch_info['model'] == 'de':
    tic = timeit.default_timer()
    loss = train_de(model, batch_info['epochs'], x_tensor, y_tensor, x_val_tensor, y_val_tensor, optimizer,
              lr_scheduler, cuda, writer, model_weights_dir, model_name)
    toc = timeit.default_timer()

wqData.pred_time = toc - tic

print(f'\nTotal training time: {wqData.pred_time}')
print(f'Final loss: {loss:.3f}\n')

# make predictions
if config['output']['predict']['train']:
    x_train_tensor, y_train_tensor = x_tensor[0:6609, :], y_tensor[0:6609, :]  # take subset of training set
if config['output']['predict']['outliers']:
    # load outlier data
    outlierData = load_data(config['data']['outlier_fpath'])
    x_outlier, y_outlier, clusters_outliers = wqData.getXY(outlierData, print_stats=True)
    x_outlier, y_outlier, _ = wqData.preprocessXY(x_outlier, y_outlier, config, saved_model_dir,
                                                  split='', plot_hist=False)
    x_outlier_tensor = torch.from_numpy(np.float32(x_outlier))
    y_outlier_tensor = torch.from_numpy(np.float32(y_outlier))

if batch_info['model'] == 'mdn':
    output_dict = predict_mdn(model, x_test_tensor, y_test, 'test', {}, cuda, mean_mixture=batch_info['mdn_mean_mixture'])
    if config['output']['predict']['train']:
        output_dict = predict_mdn(model, x_train_tensor, y_train_tensor, 'train', output_dict, cuda, mean_mixture=batch_info['mdn_mean_mixture'])
    if config['output']['predict']['outliers']:
        output_dict = predict_mdn(model, x_outlier_tensor, y_outlier_tensor, 'outliers', output_dict, cuda, mean_mixture=batch_info['mdn_mean_mixture'])

elif batch_info['model'] == 'de':
    output_dict = predict_de(model, x_test_tensor, y_test, 'test', {}, cuda)
    if config['output']['predict']['train']:
        output_dict = predict_de(model, x_train_tensor, y_train_tensor, 'train', output_dict, cuda)
    if config['output']['predict']['outliers']:
        output_dict = predict_de(model, x_outlier_tensor, y_outlier_tensor, 'outliers', output_dict, cuda)

#Save scalers and PCA to use on future evaluations
wqData.save_scalers(os.path.join(saved_model_dir, config['output']['scalers']['x_scaler_fname']),
        os.path.join(saved_model_dir, config['output']['scalers']['y_scaler_fname']))
wqData.save_pca(os.path.join(saved_model_dir, config['output']['pca_fname']))

# evaluate model
log_results = batch_info['results_as_log']
for key in output_dict:
    results = wqData.prep_results(y)
    uncertainty = output_dict[key]['uncertainty']
    _y_pred = output_dict[key]['y_pred']
    _y_test = output_dict[key]['y_test']

    if 'outliers' in key:
        clust = pd.DataFrame(clusters_outliers, columns=['cluster'])
    elif 'test' in key:
        clust = clust_test
    elif 'train' in key:
        clust = clust_train[0:6609]

    # evaluate model predictions
    _y_test, _y_pred = np.vstack(_y_test), np.vstack(_y_pred)  # stack predictions
    results = evaluate(wqData, _y_pred, uncertainty, _y_test, results, 'final', log_results)
    if config["output"]["owt"]:
        results = owt_evaluate(wqData, clust, _y_pred, uncertainty, _y_test, results, log_results)
    if config["output"]["pis"]:
        results = evaluate_pis(_y_test, _y_pred, uncertainty, results, targets, 'final', interval=0.95)
    results['batch_info'] = batch_info

    # save results as a dictionary
    f = open(os.path.join(saved_model_dir, key + '_' + model_name + '_results.pkl'), 'wb')
    pickle.dump(results, f)
    f.close()

    # create directory for plots
    plot_dir = os.path.join(saved_model_dir, key + '_' + 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # save results and plots
    if "feature_importance" in config["output"].keys():
        config["batch_info"]["cuda"] = cuda
        sensor = config['sensor_data_info']['sensor']
        x_names = sensorData[sensor].filter(regex='^[0-9]').columns
        eval_feature_importance(config, model, x_test_tensor, y_test, plot_dir, x_names)
    write_results(results, config, plot_dir, log_results) #write results and make plots

if config['output']['predict']['outliers']:
    # flatten arrays
    for key in output_dict['outliers'].keys():
        output_dict['outliers'][key] = np.squeeze(output_dict['outliers'][key])
    for key in output_dict['test'].keys():
        output_dict['test'][key] = np.squeeze(output_dict['test'][key])

    outlier_df = pd.DataFrame.from_dict(output_dict['outliers'], orient='columns')
    outlier_df['type'] = 'outlier'
    test_df = pd.DataFrame.from_dict(output_dict['test'], orient='columns')
    test_df['type'] = 'test'
    merged_outlier_df = pd.concat([outlier_df, test_df])
    results['uncertainty_eval'] = evaluate_uncertainty(merged_outlier_df, config["sensor_data_info"]["targets"][0], saved_model_dir)


# to load saved model:
# mdn_model = MDN(sensor_data.n_in, batch_info['n_hidden'], batch_info['num_gaussians'])
# mdn_model.load_state_dict(torch.load(<path to model params>))

print('DONE')