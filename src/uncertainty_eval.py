import argparse
import os.path
import torch
import numpy as np
import pandas as pd
from src.MLP_retrieval_PyTorch import WQ_Data
from src.evaluation import evaluate, owt_evaluate, evaluate_pis
from src.train_mdn import predict_mdn
from src.train_de import predict_de
from src.utils import read_yaml, load_data, update_batch_info, load_from_state, load_full_model
from src.viz_utils import write_results, examine_outliers
import matplotlib.pyplot as plt


def generate_results(data, conf, output_path):

    batch_info = conf["batch_info"]

    wq_data = WQ_Data(conf['sensor_data_info'])

    # get the X (inputs) and Y (outputs) data
    # y = log transformed and standard scaled
    # y0 = raw y values
    x, y, cluster = wq_data.getXY(data)
    cluster = pd.DataFrame(cluster, columns=['cluster'])
    x, y, _ = wq_data.preprocessXY(x, y, conf, output_path, split='train', plot_hist=False)  #TODO: read in PCA values from results.pkl

    # TODO: make this work on DE when we need to load X models
    if conf["model"]["param_fpath"]:
        model = load_from_state(conf["home_dir"] + conf["model"]["param_fpath"], wq_data.n_in, batch_info)
    else:
        model = load_full_model(conf["home_dir"] + conf["model"]["model_fpath"])

    # prep the results dictionary for the various evaluation statistics
    results = wq_data.prep_results(y)

    #test_dataloader = wq_dataloader(x, y, batch_size=batch_info['batch_size'],
    #                           shuffle=False, num_workers=0)

    x_tensor = torch.from_numpy(np.float32(x))
    y = np.float32(y)

    # evaluate model on full test set
    if batch_info['model'] == 'mdn':
        pred_dict = predict_mdn(model, x_tensor, None, None, None, conf["batch_info"]["cuda"])
    elif batch_info['model'] == 'de':
        pred_dict = predict_de(model, x_tensor, None, None, None, conf["batch_info"]["cuda"])

    y_pred = pred_dict['y_pred']
    y_true, y_pred = np.vstack(y), np.vstack(y_pred)  # stack predictions

    for unc in pred_dict.keys():
        # evaluate model predictions
        uncertainty = pred_dict[unc]['uncertainty']
        results = evaluate(wq_data, y_pred, uncertainty, y_true, results, 'final')
        results = owt_evaluate(wq_data, cluster, y_pred, uncertainty, y_true, results)

    results['batch_info'] = batch_info

    return results


def load_results(conf):
    plot_dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])
    if conf["data"]["results_fpath"]:
        results = load_data(os.path.join(conf["home_dir"], conf["data"]["results_fpath"]))
    else:
        batch_info = conf["batch_info"]
        batch_info = update_batch_info(batch_info)
        conf["batch_info"] = batch_info

        data = load_data(os.path.join(conf["home_dir"], conf["data"]["input_fpath"]))
        results = generate_results(data, conf, plot_dir)

    return results


def plot_uncertainty(results, conf):
    rmse, filtered_rmse = [], []
    cutoff = np.arange(0.5, 4, 0.1)
    for val in cutoff:
        _rmse, _filtered_rmse = examine_outliers(results, conf, unc_cutoff=val, output_path=None)
        rmse.extend([_rmse])
        filtered_rmse.extend([_filtered_rmse])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_title("RMSE vs. Uncertainty cutoff")
    ax.set_xlabel("Uncertainty cutoff")
    ax.set_ylabel("RMSE")
    ax.plot(cutoff, filtered_rmse) #, linestyle='', marker='o')
    fig.savefig('RMSE vs. Uncertainty cutoff.png', dpi=300, bbox_inches="tight")


def calc_partial_RMSE(results, conf, percent_removed, unc_cutoff, plot_hist):
    plot_dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])
    rmse, filtered_rmse = examine_outliers(results, conf, unc_percent=percent_removed,
                                           output_path=plot_dir, plot_hist=plot_hist)
    print('RMSE: ', rmse)
    print('Filtered RMSE after removing ', percent_removed * 100, '% of the data: ', filtered_rmse)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    conf = read_yaml(args.yaml)
    results = load_results(conf)

    plot_uncertainty(results, conf)  # plot histogram of uncertainties
    percent_removed = 0.15  # percentage of data to remove
    unc_cutoff = 0.99  # when plotting uncertainty hist. remove extreme data
    calc_partial_RMSE(results, conf, percent_removed, unc_cutoff, plot_hist=True)
