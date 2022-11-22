import yaml
import torch
import pickle
import os
import numpy as np
import pandas as pd
from src.mdn_model import MDN
from src.de_model import DE

def read_csv_data(fpath, sensor, output_file):
    #Read data and format so it can be used with pre-existing code
    data = pd.read_csv(fpath) #.to_dict()

    if "Chla" in data.keys():
        data["chl"] = data["Chla"]    
    #data["cluster"] = np.zeros((data["chl"].shape)) - 1.0
    
    data = {sensor: data} 

    #Output in expected format for easy use in pre-existing pipeline
    if output_file is not None:
        with open(output_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def read_yaml(fpath_yaml):
    yml_conf = None
    with open(fpath_yaml) as f_yaml:
        yml_conf = yaml.load(f_yaml, Loader=yaml.FullLoader)
    return yml_conf


def load_data(data_path, sensor="hico"):
    if os.path.splitext(data_path)[1] == ".csv":
        return read_csv_data(data_path, sensor, None)
    else:
        with open(data_path, 'rb') as fp:
            data = pickle.load(fp)
        return data


def load_full_model(fpath):
    model = torch.load(fpath)
    model.eval()
    return model


def update_batch_info(batch_info):
    for key, value in batch_info.items():
        value = None if value == 'None' else value
        batch_info[key] = value

    batch_info['cuda'] = torch.cuda.is_available()
 
    return batch_info


def load_from_state(fpath, mdl, n_in, batch_info, strict=False):
    n_hidden = batch_info["n_hidden"]
    num_gaussians = batch_info["num_gaussians"]
    num_lin_layers = batch_info["num_lin_layers"]
    if mdl == "mdn":
        model = MDN(n_in, n_hidden, num_gaussians, num_lin_layers)
        model.load_state_dict(torch.load(fpath[0]), strict=strict)
    else:
        model = DE(num_models=num_gaussians, inputs=n_in, hidden_layers=n_hidden, n_lin_layers=num_lin_layers)
        for i in range(len(fpath)):
            ensemble_mem = getattr(model, 'model_' + str(i))
            ensemble_mem.load_state_dict(torch.load(fpath[i]), strict=strict)
    model.eval()
    return model



    






