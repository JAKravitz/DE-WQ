import argparse
import torch
import os
import numpy as np
import pandas as pd
from captum.attr import FeaturePermutation
from src.MLP_retrieval_PyTorch import WQ_Data
from src.dataloader import wq_dataloader
from src.evaluation import evaluate, owt_evaluate, evaluate_pis
from src.utils import read_yaml, load_data, update_batch_info, load_from_state, load_full_model, read_csv_data
from src.viz_utils import write_results, generate_feature_importance_plot
from src.scorers import get_scorer


def generate_results(data, conf, output_path=None, log=True):
    
    batch_info = conf["batch_info"]

    wq_data = WQ_Data(conf['sensor_data_info'])

    # get the X (inputs) and Y (outputs) data
    # y = log transformed and standard scaled
    # y0 = raw y values
    model_dir = os.path.join(conf['home_dir'], conf['output']['model_out_dir'])
    wq_data.load_scalers(os.path.join(model_dir, conf['output']['scalers']['x_scaler_fname']),
        os.path.join(model_dir, conf['output']['scalers']['y_scaler_fname']))
    x, y, cluster = wq_data.getXY(data)
    cluster = pd.DataFrame(cluster, columns=['cluster'])
    x, y, _ = wq_data.preprocessXY(x, y, conf, model_dir, split='val', plot_hist=True)


    sensor = conf["sensor_data_info"]["sensor"]
    x_names = data[sensor].filter(regex='^[0-9]').columns 

    print(wq_data.n_in)
    #Used to load model from params 
    if conf["model"]["param_fpath"]:
        params = []
        if type(conf["model"]["param_fpath"]) == list:
            for i in range(len(conf["model"]["param_fpath"])):
                params.append(os.path.join(conf["home_dir"], conf["model"]["param_fpath"][i]))
        else:
            params = [os.path.join(conf["home_dir"], conf["model"]["param_fpath"])]
        model = load_from_state(params, batch_info["model"], wq_data.n_in, batch_info)
    #Used to load model from serialized object
    else:
        model = load_full_model(os.path.join(conf["home_dir"], conf["model"]["model_fpath"]))

    # prep the results dictionary for the various evaluation statistics
    results = wq_data.prep_results(y)
 
    #test_dataloader = wq_dataloader(x, y, batch_size=batch_info['batch_size'],
    #                           shuffle=False, num_workers=0)

    x_tensor = torch.from_numpy(np.float32(x))

    if "feature_importance" in conf["output"].keys():
        eval_feature_importance(conf, model, x_tensor, y, output_path, x_names)

    y = np.float32(y)    

    if conf['batch_info']['model'] == 'mdn':
        from src.train_mdn import predict
    elif conf['batch_info']['model'] == 'de':
        from src.train_de import predict

    # evaluate model on full test set
    if conf['batch_info']['model'] == 'mdn':
        y_pred, al_unc, ep_unc = predict(model, x_tensor, conf["batch_info"]["cuda"])
        unc = np.sqrt(al_unc + ep_unc) 
    else:
        y_pred, unc = predict(model, x_tensor, conf["batch_info"]["cuda"])

    # evaluate model predictions
    y_true, y_pred = np.vstack(y), np.vstack(y_pred)  # stack predictions
    results = evaluate(wq_data, y_pred, unc, y_true, results, 'final', log)
    if conf["output"]["owt"]:
        results = owt_evaluate(wq_data, cluster, y_pred, unc, y_true, results, log)

    # calculate prediction interval coverage probability (PICP) and mean prediction
    # interval width (MPIW)
    if conf["output"]["pis"]:
        results = evaluate_pis(y_true, y_pred, unc, results, conf["sensor_data_info"]["targets"], 'final', interval=0.95)
 
    results['batch_info'] = batch_info
 
    return results 


def eval_feature_importance(conf, model, x, y, output_path=None, feature_names = None):

    if conf['batch_info']['model'] == 'mdn':
        from src.train_mdn import predict
    elif conf['batch_info']['model'] == 'de':
        from src.train_de import predict

    scores = conf["output"]["feature_importance"]["scores"]

    for score in scores:
        scorer = get_scorer(score)
        if conf['batch_info']['model'] == 'mdn':
            forward_func = lambda x, y, predict=predict, model=model, scorer=scorer, conf=conf : torch.as_tensor(scorer(predict(model, x, conf["batch_info"]["cuda"], conf["batch_info"]["mdn_mean_mixture"])[0], y))
        elif conf['batch_info']['model'] == 'de':
            forward_func = lambda x, y, predict=predict, model=model, scorer=scorer, conf=conf : torch.as_tensor(scorer(predict(model, x, conf["batch_info"]["cuda"])[0], y))
        feature_perm = FeaturePermutation(forward_func)
        attr = feature_perm.attribute(x, 
            show_progress=True,
            additional_forward_args=y)

        generate_feature_importance_plot(conf, attr, output_path, feature_names)


def run_test(conf, log=True):

    for i in range(len(conf["output"]["results_subdirs"])):
        
        output_path = os.path.join(conf["home_dir"], os.path.join(conf["output"]["out_dir"], conf["output"]["results_subdirs"][i]))
        print(output_path)
        os.makedirs(output_path, exist_ok=True)    

        if conf["data"]["results_fpaths"]:
            fname = os.path.join(conf["home_dir"], conf["data"]["results_fpaths"][i])
            results = load_data(fname)
        else:
            batch_info = conf["batch_info"]
            batch_info = update_batch_info(batch_info)
            conf["batch_info"] = batch_info
 
            fname = os.path.join(conf["home_dir"],conf["data"]["eval_fpaths"][i])
            if os.path.splitext(fname)[1] == ".csv":
                #Run this one time, save for later
                out_fname = fname + ".pkl"
                read_csv_data(fname, conf["sensor_data_info"]["sensor"], out_fname)
            else:
                out_fname = fname
            data = load_data(out_fname)

            results = generate_results(data, conf, output_path, log)        

        write_results(results, conf, output_path, log)

print('DONE')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    conf = read_yaml(args.yaml)
    run_test(conf)
