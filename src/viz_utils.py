from pprint import pprint
from operator import itemgetter
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import src.scorers as sc
import math


def write_results(results, conf, output_path=None, log=True):
    print("Generating Plots")
    generate_plots(results, conf, output_path, log)
    print("Generating Results txt")
    results_to_text(results, conf, output_path)
    #examine_outliers(results, conf, unc_cutoff=None, unc_percent = 0.01, output_path=None)


def results_to_text(results, conf, output_path):
    if output_path is not None:
        out_dir = output_path
    else:
        out_dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])

    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

    out_fpath = os.path.join(conf["home_dir"], out_dir, "Result_Metrics.txt")
    with open(out_fpath, "w") as out_file:
        for tgt in targets:
            del (results[tgt]['final']['yhat'])
            del (results[tgt]['final']['ytest'])
            del (results[tgt]['final']['uncertainty'])

            for wt in results[tgt]['owt'].keys():
                del (results[tgt]['owt'][wt]['yhat'])
                del (results[tgt]['owt'][wt]['ytest'])
                del (results[tgt]['owt'][wt]['uncertainty'])

            out_file.write(tgt + ": " + os.linesep)
            pprint(results[tgt], out_file)
            out_file.write(os.linesep + os.linesep)


def generate_plots(results, conf, output_path, log):
    #print("Generating Radar Plots")
    #generate_radar_plot(results, conf, output_path)
    print("Generating Pairwise CCORR Box Plots")
    generate_pairwise_target_ccorr(results, conf, output_path)
    print("Generating Histograms")
    generate_target_histograms(results, conf, log, output_path)
    print("Generating Scatterplots")
    scatter_dens_wrap(results, conf, log, output_path)
    #print("Generating Scatterplots with Error bars")
    #generate_error_bars(results, conf, output_path)
    #print("Generating Scatterplots of Residuals vs. Uncertainties")
    #generate_residual_vs_uncertainty(results, conf, output_path)
    print("Generating ECE Plots")
    ece_wrap(results, conf, output_path)
    print("Generating Bar Plots of Computed Metrics per Water Type")
    plot_bar_metric_by_water_type(results, conf, log, output_path)
    print("Generating Box Plots of Computed Metrics per Water Type")
    plot_box_error_by_water_type(results, conf, log, output_path)
    print("Generating Uncertainty Box Plots per Water Type") 
    plot_box_uncertainty_by_water_type(results, conf, log, output_path)
 
# Only really helpful in multi-target cases
def generate_radar_plot(results, conf, output_path):
    for evl in conf["output"]["evaluations"]:
        targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

        final_rad = 2 * np.pi
        if len(targets) == 3:
            final_rad = np.pi
        elif len(targets) == 2:
            final_rad = np.pi

        num_results = len(results[targets[0]][evl][conf["output"]["radar"]["metrics"][0]])

        ax = None
        fig = None
        for metric in conf["output"]["radar"]["metrics"]:
            i = -1
            data_full = {}
            tgt_final = []
            label_loc = None
            for result in range(num_results):
                i += 1
                data = []
                for tgt in targets:
                    data.append(results[tgt][evl][metric][result])

                if i == 0:
                    label_loc = np.linspace(start=0, stop=np.pi, num=len(data))
                    fig = plt.figure(figsize=(5, 5))
                    ax = fig.add_subplot(111, projection='polar')
                if num_results > 1:
                    key = metric + "_result" + str(result)
                else:
                    key = metric
                data_full[key] = data

            for metric2 in data_full.keys():
                ax.plot(label_loc, data_full[metric2], label=metric2)
            ax.set_title('Metric Comparison', size=20)
            lines, labels = ax.set_thetagrids(np.degrees(label_loc), labels=targets)
            ax.legend()
            if output_path is not None:
                fig.savefig(os.path.join(output_path, "Radar_Plot_" + metric + "_" + evl + ".png"), 
                    dpi=300, bbox_inches="tight")
            else:
                fig.savefig(os.path.join(conf["home_dir"], conf["output"]["out_dir"], "Radar_Plot_" + metric + "_" + evl + ".png"),
                    dpi=300, bbox_inches="tight")
            plt.clf()
    plt.close()

def generate_pairwise_target_ccorr(results, conf, output_path):
    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))
 
    for evl in conf["output"]["evaluations"]:
        pred_data, pred_keys = generate_ccorr(results, targets, evl, y_hat=True)
        true_data, true_keys = generate_ccorr(results, targets, evl)

        ax = None
        fig = None
        for i in range(len(pred_keys)):
            if i == 0:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)   

            true_index = np.flatnonzero(np.core.defchararray.find(true_keys, pred_keys[i]) != -1)[0]

            data = [pred_data[i], true_data[true_index]]
            keys = ["", "Predicted", "True"]
            bp = ax.boxplot(data)
            ax.set_xticks(np.arange(3), keys)
            if output_path is not None:
                fig.savefig(os.path.join(output_path, "CCORR_" + pred_keys[i] + "_" + evl + ".png"),
                    dpi=300, bbox_inches="tight")
            else:
                fig.savefig(os.path.join(conf["home_dir"], conf["output"]["out_dir"], "CCORR_" + pred_keys[i] + "_" + evl + ".png"),
                    dpi=300, bbox_inches="tight")
            plt.clf()
    plt.close()

def generate_ccorr(results, targets, evl, y_hat=False):
    ccorr_dict = {}
    ccorr_data = []
    ccorr_keys = []

    for tgt in targets:
        for tgt2 in targets:
            if tgt in ccorr_dict.keys():
                if tgt2 in ccorr_dict[tgt]:
                    continue
                else:
                    ccorr_dict[tgt].append(tgt2)
            else:
                ccorr_dict[tgt] = [tgt2]

            if tgt2 in ccorr_dict.keys():
                if tgt != tgt2 and tgt in ccorr_dict[tgt2]:
                    continue
                else:
                    ccorr_dict[tgt2].append(tgt)
            else:
                ccorr_dict[tgt2] = [tgt]

            key = tgt + "_" + tgt2
            y_type = "ytest"
            if y_hat:
                y_type = "yhat"

            # For each result
            for result in range(len(results[tgt][evl][y_type])):
                for result2 in range(len(results[tgt2][evl][y_type])):
                    if tgt2 == tgt and result == result2:
                        continue
                    if len(results[tgt][evl][y_type]) > 1 or len(results[tgt2][evl][y_type]) > 1:
                        key = tgt + str(result) + "_" + tgt2 + str(result2)
                    else:
                        key = tgt + "_" + tgt2
                    ccorr_data.append(np.correlate(results[tgt][evl][y_type][result].to_numpy(),
                                                   results[tgt2][evl][y_type][result2].to_numpy(), "same"))
                    ccorr_keys.append(key)

    return ccorr_data, ccorr_keys


def generate_target_histograms(results, conf, log, output_path=None):
    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

    for evl in conf["output"]["evaluations"]:
        for tgt in targets:
            ax = None
            fig = None
            for result in range(len(results[tgt][evl]['yhat'])):

                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)

                data = results[tgt][evl]['yhat'][result].to_numpy()
                label = tgt + "_pred"
                if len(results[tgt][evl]['yhat']) > 1:
                    label = tgt + "_result" + str(result) + "_pred"
                ax.hist(data, label=label, alpha=0.4, color="orange", bins=40)

                data = results[tgt][evl]['ytest'][result].to_numpy()
                label = tgt
                if len(results[tgt][evl]['ytest']) > 1:
                    label = tgt + "_result" + str(result)
                ax.hist(data, label=label, alpha=0.4, color="blue", bins=40)

                if log:
                    ax.set_xlabel(f"log({tgt} concentration) (mg/m^3)")
                else:
                    ax.set_xlabel(f"{tgt} concentration (mg/m^3)")
                ax.set_ylabel("counts")
                ax.set_title(f"Histogram of {tgt} concentration")
                ax.legend()

                if output_path is not None:
                    fname = os.path.join(output_path, "Hist_" + tgt)
                else:
                    fname = os.path.join(conf["home_dir"], conf["output"]["out_dir"], "Hist_" + tgt)
                if len(results[tgt][evl]['yhat']) > 1:
                    fname += "_result" + str(result)
                fname += "_" + evl + ".png"
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.clf()
    plt.close()


def scatter_dens_wrap(results, conf, log, output_path=None):
    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

    # directory for saving plots
    if output_path is not None:
        dir = output_path
    else:
        dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])

    for evl in conf["output"]["evaluations"]:
        for tgt in targets:
            for result in range(len(results[tgt][evl]['yhat'])):
                pred = results[tgt][evl]['yhat'][result].to_numpy()
                truth = results[tgt][evl]['ytest'][result].to_numpy()
                unc = results[tgt][evl]['uncertainty'][result].to_numpy()

                title = "Scatter_" + tgt
                x_name = "y"
                y_name = "yhat"
                scatter_density(truth, pred, x_name, y_name, title, dir, log, save_IO=True)

                title = "Scatter_Uncertainty_" + tgt
                x_name = "yhat"
                y_name = "uncertainty"
                scatter_density(pred, unc, x_name, y_name, title, dir, log, save_IO=True)

    for tgt in targets:
        clusters = results[tgt]['owt'].keys()
        for clst in clusters:
            for result in range(len(results[tgt]['owt'][clst]['yhat'])):
                pred = results[tgt]['owt'][clst]['yhat'][result].to_numpy()
                truth = results[tgt]['owt'][clst]['ytest'][result].to_numpy()
                unc = results[tgt]['owt'][clst]['uncertainty'][result].to_numpy()

                title = "Scatter_" + clst + "_Subset_" + tgt
                #print(clst, truth.shape, pred.shape)
                x_name = "y"
                y_name = "yhat"
                scatter_density(truth, pred, x_name, y_name, title, dir, log, save_IO=True)

                title = "Scatter_Uncertainty_" + clst + "_Subset_" + tgt
                x_name = "yhat"
                y_name = "uncertainty"
                scatter_density(pred, unc, x_name, y_name, title, dir, log, save_IO=True)


def scatter_density(x, y, x_name, y_name, title, dir, log, save_IO=False):
    '''
    makes a scatter plot and color codes where most of the data is
    :param x: x-value
    :param y: y-value
    :param x_name: name on x-axis
    :param y_name: name on y-axis
    :param title: title
    :param dir: save location
    :param save_IO: save plot?
    :return: -
    '''
    # print('making scatter density plot ... ')
    # need to reduce number of samples to keep processing time reasonable.
    # Reduce if processing time too long or run out of RAM
    max_n = 50000
    if len(x) > max_n:
        subsample = int(len(x) / max_n)
        x = x[::subsample]
        y = y[::subsample]
    try:
        r, _ = stats.pearsonr(x, y)  # get R
    except:
        print('could not calculate r, set to nan')
        r = np.nan
    xy = np.vstack([x, y])
    # test if x == y  # test if x or y don't have a range of values. Otherwise stats.gaussian_kde throws error
    if (np.mean(x) == np.mean(y)) or (np.min(x) == np.max(x)) or (np.min(y) == np.max(y)):
        z = np.arange(len(x))
    else:
        z = stats.gaussian_kde(xy)(xy)  # calculate density
    # sort points by density
    idx = z.argsort()
    d_feature = x[idx]
    d_target = y[idx]
    z = z[idx]
    # plot everything
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
 
    ax.scatter(d_feature, d_target, c=z, s=2, label='R = ' + str(np.round(r, 2)))
    ax.legend()
    ax.set_xlim(np.percentile(d_feature, 1), np.percentile(d_feature, 99))
    ax.set_ylim(np.percentile(d_target, 1), np.percentile(d_target, 99))
    if log:
        plot_x_name = "log(" + x_name + ") (mg/m^3)"
        plot_y_name = "log(" + y_name + ") (mg/m^3)"
    else:
        plot_x_name = x_name + " (mg/m^3)"
        plot_y_name = y_name + " (mg/m^3)"
    ax.set_ylabel(plot_y_name)
    ax.set_xlabel(plot_x_name)
    # plt.ylim([-2, 2])
    ax.set_title(title)
    plt.tight_layout()

    if save_IO:
        # save to file
        fig_dir = os.path.join(dir, title + "_" + x_name + "_" + y_name + '.png')
        fig.savefig(fig_dir, dpi=300, bbox_inches="tight")
    else:
        fig.show()
    plt.close()
    

def generate_error_bars(results, conf, output_path):
    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

    for evl in conf["output"]["evaluations"]:
        for tgt in targets:
            for result in range(len(results[tgt][evl]['yhat'])):
                pred = results[tgt][evl]['yhat'][result].to_numpy()
                truth = results[tgt][evl]['ytest'][result].to_numpy()
                unc = results[tgt][evl]['uncertainty'][result].to_numpy()
                df = pd.DataFrame(np.column_stack((truth, pred, unc)), columns=['truth', 'pred', 'unc'])

                # plot all test points
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                ax.errorbar(truth, pred, yerr=unc, fmt='o')
                ax.set_ylabel('y_hat')
                ax.set_xlabel('y')
                ax.set_title('Scatter_' + tgt + ' w/error bars')

                if output_path is not None:
                    fname = os.path.join(output_path, 'Error_bars_' + tgt)
                else:
                    fname = os.path.join(conf['home_dir'], conf['output']['out_dir'], 'Error_bars_' + tgt + '_ramdom_1k')
                if len(results[tgt][evl]['yhat']) > 1:
                    fname += '_result' + str(result)
                fname += '_' + evl + '.png'
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.clf()

                # take a random sample
                df_random = df.sample(1000)
                truth_random = df_random[['truth']].values.T[0]
                pred_random = df_random[['pred']].values.T[0]
                unc_random = df_random[['unc']].values.T[0]
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                ax.errorbar(truth_random, pred_random, yerr=unc_random, fmt='o')
                ax.set_ylabel('y_hat')
                ax.set_xlabel('y')
                ax.set_title('Scatter_' + tgt + ' w/error bars - random subset')

                if output_path is not None:
                    fname = os.path.join(output_path, 'Error_bars_' + tgt + '_ramdom_1k')
                else:
                    fname = os.path.join(conf['home_dir'], conf['output']['out_dir'], 'Error_bars_' + tgt + '_ramdom_1k')
                if len(results[tgt][evl]['yhat']) > 1:
                    fname += '_result' + str(result)
                fname += '_' + evl + '.png'
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.clf()

                # sort in decreasing order of uncertainty
                df = df.sort_values(by=['unc'], ascending=False)
                df = df.head(1000)
                print(df.head())
                truth_sorted = df[['truth']].values.T[0]
                pred_sorted = df[['pred']].values.T[0]
                unc_sorted = df[['unc']].values.T[0]
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                ax.errorbar(truth_sorted, pred_sorted, yerr=unc_sorted, fmt='o')
                ax.set_ylabel('y_hat')
                ax.set_xlabel('y')
                ax.set_title('Scatter_' + tgt + ' w/error bars - top 1k uncertainties')

                if output_path is not None:
                    fname = os.path.join(output_path, 'Error_bars_' + tgt + '_top_1k_uncertainties')
                else:
                    fname = os.path.join(conf['home_dir'], conf['output']['out_dir'], 'Error_bars_' + tgt + '_top_1k_uncertainties')
                if len(results[tgt][evl]['yhat']) > 1:
                    fname += '_result' + str(result)
                fname += '_' + evl + '.png'
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.clf()
    plt.close()

def generate_residual_vs_uncertainty(results, conf, output_path):
    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

    for evl in conf["output"]["evaluations"]:
        for tgt in targets:
            for result in range(len(results[tgt][evl]['yhat'])):
                pred = results[tgt][evl]['yhat'][result].to_numpy()
                truth = results[tgt][evl]['ytest'][result].to_numpy()
                unc = results[tgt][evl]['uncertainty'][result].to_numpy()
                residuals = abs(pred - truth)
                df = pd.DataFrame(np.column_stack((truth, pred, unc, residuals)),
                                  columns=['truth', 'pred', 'unc', 'residuals'])

               # plot residuals vs unc
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                ax.plot(residuals, unc, linestyle='', marker='o')
                ax.set_ylabel('uncertainties')
                ax.set_xlabel('residuals')
                ax.set_title('Residuals vs. uncertainties_' + tgt)

                if output_path is not None:
                    fname = os.path.join(output_path, 'Residuals_vs_uncertainties_' + tgt)
                else:
                    fname = os.path.join(conf['home_dir'], conf['output']['out_dir'],
                                         'Residuals_vs_uncertainties_' + tgt)
                if len(results[tgt][evl]['yhat']) > 1:
                    fname += '_result' + str(result)
                fname += '_' + evl + '.png'
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.clf()

                # take a random sample
                df_random = df.sample(1000)
                residuals_random = df_random[['residuals']].values.T[0]
                unc_random = df_random[['unc']].values.T[0]
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                ax.plot(residuals_random, unc_random, linestyle='', marker='o')
                ax.set_ylabel('uncertainties')
                ax.set_xlabel('residuals')
                ax.set_title('Residuals vs. uncertainties_' + tgt + ' - random 1k')

                if output_path is not None:
                    fname = os.path.join(output_path, 'Residuals_vs_uncertainties_' + tgt + '_random_1k')
                else:
                    fname = os.path.join(conf['home_dir'], conf['output']['out_dir'],
                                         'Residuals_vs_uncertainties_' + tgt + '_random_1k')
                if len(results[tgt][evl]['yhat']) > 1:
                    fname += '_result' + str(result)
                fname += '_' + evl + '.png'
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.clf()

                # sort in decreasing order of uncertainty
                df = df.sort_values(by=['unc'], ascending=False)
                df = df.head(1000)
                print(df.head())
                residuals_sorted = df[['residuals']].values.T[0]
                unc_sorted = df[['unc']].values.T[0]
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                ax.plot(residuals_sorted, unc_sorted, linestyle='', marker='o')
                ax.set_ylabel('uncertainties')
                ax.set_xlabel('residuals')
                ax.set_title('Residuals vs. uncertainties_' + tgt + ' - top 1k uncertainties')

                if output_path is not None:
                    fname = os.path.join(output_path, 'Residuals_vs_uncertainties_' + tgt + '_top_1k_uncertainties')
                else:
                    fname = os.path.join(conf['home_dir'], conf['output']['out_dir'], 'Residuals_vs_uncertainties_' + tgt + '_top_1k_uncertainties')
                if len(results[tgt][evl]['yhat']) > 1:
                    fname += '_result' + str(result)
                fname += '_' + evl + '.png'
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.clf()
    plt.close()


def ece_wrap(results, conf, output_path):
    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

    # directory for saving plots
    if output_path is not None:
        dir = output_path
    else:
        dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])

    sigmas = conf["output"]["ece"]["sigma_bins"]

    for evl in conf["output"]["evaluations"]:
        for tgt in targets:
            for result in range(len(results[tgt][evl]['yhat'])):
                pred = np.squeeze(results[tgt][evl]['yhat'][result].to_numpy())
                truth = np.squeeze(results[tgt][evl]['ytest'][result].to_numpy())
                unc = np.squeeze(results[tgt][evl]['uncertainty'][result].to_numpy())

                title = "ECE_" + tgt
                plot_ece(truth - pred, unc, sigmas, title, dir)


    for tgt in targets:
        clusters = results[tgt]['owt'].keys()
        for clst in clusters:
            for result in range(len(results[tgt]['owt'][clst]['yhat'])):
                pred = results[tgt]['owt'][clst]['yhat'][result].to_numpy()
                truth = results[tgt]['owt'][clst]['ytest'][result].to_numpy()
                unc = results[tgt]['owt'][clst]['uncertainty'][result].to_numpy()

                title = "ECE_" + clst + "_Subset_" + tgt
                plot_ece(truth - pred, unc, sigmas, title, dir)



def plot_ece(res_diff,y_uncert,sigmas,title,dir):
    expected_fraction = [0.0]
    obs_fraction = [0.0]

    # calculate fraction for each sigma coeff.
    for s in sigmas:
        sigma_upper = s*y_uncert.sum()/y_uncert.shape
        sigma_lower = sigma_upper*-1
        calibration_vector = np.zeros(y_uncert.shape)
        for idx, y in enumerate(res_diff):
            #print(y)
            if y <= sigma_upper and y >= sigma_lower:
                calibration_vector[idx] = 1
        tol_interval = calibration_vector.sum()/calibration_vector.shape
        #print(tol_interval)
        obs_fraction.append(tol_interval)
        percent_obs_within_s = round(1 - (1 - stats.norm.cdf(s)) * 2,3)
        expected_fraction.append(percent_obs_within_s)
    obs_fraction.append(1.0)
    expected_fraction.append(1.0)

    # make plot
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.plot(expected_fraction,obs_fraction, linestyle='--', lw = 2, color = 'orange', alpha = 0.5)
    ax.scatter(expected_fraction,obs_fraction)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    xpoints = ypoints = plt.xlim()
    ax.plot(xpoints, ypoints, linestyle='-', color='k', lw=1.5, scalex=False, scaley=False, alpha = 0.7, zorder = 0)
    ax.set_ylabel("Observed Fraction")
    ax.set_xlabel("Expected Fraction")
    fig.savefig(os.path.join(dir,title+".png"), dpi = 300, bbox_inches="tight")
    plt.clf()
    plt.close()


def generate_feature_importance_plot(conf, attr, output_path=None, feature_names = None):

    metrics = conf["output"]["feature_importance"]["scores"]

    # directory for saving plots
    if output_path is not None:
        dir = output_path
    else:
        dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])

    for i in range(len(metrics)):
 
        metric = metrics[i]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        ax.bar(range(attr.shape[1]), np.squeeze(attr[i].numpy()), align='center')
        ax.set_xlabel("Features")
        ax.set_ylabel("Attributions")
        ax.set_title("Feature Importance Using " + metric)

        if feature_names is not None:
            ax.set_xticks(np.arange(len(feature_names)), labels=feature_names, rotation=90, fontsize=4)

        fig.savefig(os.path.join(dir,"Feature_Importance_" + metric + "_.png"), dpi=300, bbox_inches="tight")
        plt.clf()
    plt.close()


def plot_bar_metric_by_water_type(results, conf, log, output_path=None):

    # directory for saving plots
    if output_path is not None:
        dir = output_path
    else:
        dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])

    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))
 
    for tgt in targets:
        if tgt in ["fit_time", "pred_time", "train_loss", "val_loss", "batch_info"]:
            continue
        for metric in results[tgt]['final'].keys():
            if metric in ["ytest", "yhat", "uncertainty", "PICP", "MPIW"]:
                continue

            data = [results[tgt]['final'][metric][0]]
            labels = ['Total']

            for key in results[tgt]['owt'].keys():
                if key.isnumeric(): #exclude clusters (for now, at least)
                    continue
                labels.append(key)
                if results[tgt]['owt'][key][metric]:  # only append if list is not empty
                    data.append(results[tgt]['owt'][key][metric][0])
                else:
                    data.append(0)

            fig_x = len(labels) / 8 * 5  # scale plot size based on # of keys to plot
            fig = plt.figure(figsize=(fig_x, 5))
            ax = fig.add_subplot(111)
            ax.bar(range(len(labels)), data, align='center', tick_label = labels)
            ax.set_ylabel(metric + (" (mg/m^3)"))
            ax.set_xlabel("Water Type")
            if log:
                ax.set_title("log(" + tgt + ") " + metric + " by water type")
            else:
                ax.set_title(tgt + " " + metric + " by water type")
            fig.savefig(os.path.join(dir, tgt + "_" + metric + "_bar.png"), dpi=300, bbox_inches="tight")
            plt.clf()

    plt.close()


def plot_box_uncertainty_by_water_type(results, conf, log, output_path=None):

    # directory for saving plots
    if output_path is not None:
        dir = output_path
    else:
        dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])

    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

    for tgt in targets:
        data = [results[tgt]['final']['uncertainty'][0].to_numpy()]
        labels = ['Total']
    
        for key in results[tgt]['owt'].keys():
            if key.isnumeric(): #exclude clusters (for now, at least)
                continue
            labels.append(key)
            if results[tgt]['owt'][key]['uncertainty']:  # only append if list is not empty
                data.append(results[tgt]['owt'][key]['uncertainty'][0].to_numpy())
            else:
                data.append(0)

        fig_x = len(labels) / 8 * 5  # scale plot size based on # of keys to plot
        fig = plt.figure(figsize=(fig_x, 5))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data)
        ax.set_xticks(range(1, len(labels)+1), labels)
        ax.set_ylabel("Uncertainty (mg/m^3)")
        ax.set_xlabel("Water Type")
        if log:
            ax.set_title("log(" + tgt + ") uncertainty by water type")
        else:
            ax.set_title(tgt + " uncertainty by water type")
        fig.savefig(os.path.join(dir, tgt + "_uncertainty_box.png"), dpi=300, bbox_inches="tight")
        plt.clf() 
    plt.close()


def plot_box_error_by_water_type(results, conf, log, output_path=None):
    # same as box plot for uncertainty but for the error of each prediction

    # directory for saving plots
    if output_path is not None:
        dir = output_path
    else:
        dir = os.path.join(conf["home_dir"], conf["output"]["out_dir"])

    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))

    for tgt in targets:
        # calculate error for each predicted sample
        data = [results[tgt]['final']['yhat'][0].to_numpy() - results[tgt]['final']['ytest'][0].to_numpy()]
        labels = ['Total']

        for key in results[tgt]['owt'].keys():
            if key.isnumeric():  # exclude clusters (for now, at least)
                continue
            labels.append(key)
            if results[tgt]['owt'][key]['yhat']:  # only append if list is not empty
                data.append(results[tgt]['owt'][key]['yhat'][0].to_numpy() - results[tgt]['owt'][key]['ytest'][0].to_numpy())
            else:
                data.append(0)

        fig_x = len(labels) / 8 * 5  # scale plot size based on # of keys to plot
        fig = plt.figure(figsize=(fig_x, 5))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_ylabel('Error' + (" (mg/m^3)"))
        ax.set_xlabel("Water Type")
        if log:
            ax.set_title("log(" + tgt + ") " + " error by water type")
        else:
            ax.set_title(tgt + " " + " error by water type")
        fig.savefig(os.path.join(dir, tgt + "_" + "error_box.png"), dpi=300, bbox_inches="tight")
        plt.clf()
    plt.close()

def examine_outliers(results, conf, unc_percent, output_path=None, plot_hist=False):

    scoreDict = {'R2': sc.r2,
                 'RMSE': sc.rmse,
                 'RMSELE': sc.rmsele,
                 'Bias': sc.bias,
                 'MAPE': sc.mape,
                 'rRMSE': sc.rrmse, }

    targets = list(set(conf["sensor_data_info"]["targets"]).intersection(results.keys()))
    if len(targets) == 0:
        raise Exception("Empty targets list. sensor_data_info['targets'] is likely misspecified.")

    metrics = {'R2': [],
               'RMSE': [],
               'RMSELE': [],
               'Bias': [],
               'MAPE': [],
               'rRMSE': []
                }
    filtered_metrics = {'R2': [],
                       'RMSE': [],
                       'RMSELE': [],
                       'Bias': [],
                       'MAPE': [],
                       'rRMSE': []
                        }

    for evl in conf["output"]["evaluations"]:
        for tgt in targets:
            for result in range(len(results[tgt][evl]['yhat'])):
                pred = results[tgt][evl]['yhat'][result].to_numpy()
                truth = results[tgt][evl]['ytest'][result].to_numpy()
                unc = results[tgt][evl]['uncertainty'][result].to_numpy()
                residuals = abs(np.exp(pred) - np.exp(truth))


                """print('\nUncertainty')
                print('\tmax: ', np.max(unc))
                print('\tsd: ', np.mean(unc))
                print('Residuals')
                print('\tmean: ', np.mean(residuals))
                print('\tsd: ', np.std(residuals))
                print('\tmin: ', np.min(residuals))
                print('\tmax: ', np.max(residuals))"""

                for stat in scoreDict:
                    metrics[stat].append(scoreDict[stat](truth, pred))

                df = pd.DataFrame(np.column_stack((truth, pred, unc, residuals)),
                                  columns=['truth', 'pred', 'unc', 'residuals'])

                # plot uncertainties
                if plot_hist:
                    fig = plt.figure(figsize=(5, 5))
                    ax = fig.add_subplot(111)
                    ax.set_title("Histogram of chl uncertainties")
                    ax.set_xlabel("log(uncertainty) (mg/m^3)")
                    ax.set_ylabel("counts")
                    sorted_df = df.sort_values(by=['unc'], ascending=True)
                    data_to_keep = 1.0  # set to ~0.99 to remove 0.01 of highest uncertainties for plot
                    percentage_to_keep = int(round(data_to_keep * df.shape[0]))
                    sorted_df = sorted_df[0:percentage_to_keep]
                    to_plot = sorted_df[["unc"]].to_numpy()
                    ax.hist(to_plot, bins=30)
                    output_path
                    fig.savefig(os.path.join(output_path, 'HistogramOfUncertainties.png'),
                                dpi=300, bbox_inches="tight")

                unc_percent_to_keep = 1.0 - unc_percent
                cutoff_val = np.quantile(df['unc'].values, unc_percent_to_keep)
                df_filtered = df.loc[(df['unc'] <= cutoff_val)]
                truth_filtered = df_filtered[['truth']].values.T[0]
                pred_filtered = df_filtered[['pred']].values.T[0]


                """
                unc_filtered = df_filtered[['unc']].values.T[0]
                unc_residuals = df_filtered[['residuals']].values.T[0]
                print('\nFiltered uncertainty')
                print('\tmean: ', np.mean(unc_filtered))
                print('\tsd: ', np.std(unc_filtered))
                print('Filtered Residuals')
                print('\tmean: ', np.mean(unc_residuals))
                print('\tsd: ', np.std(unc_residuals))"""

                # plot uncertainties
                if plot_hist:
                    fig = plt.figure(figsize=(5, 5))
                    ax = fig.add_subplot(111)
                    ax.set_title("Histogram of chl uncertainties")
                    ax.set_xlabel("log(uncertainty) (mg/m^3)")
                    ax.set_ylabel("counts")
                    ax.hist(df_filtered[['unc']].values.T[0], bins=30)
                    fig.savefig(os.path.join(output_path, 'HistogramOfFilteredUncertainties.png'),
                                dpi=300, bbox_inches="tight")

                for stat in scoreDict:
                    filtered_metrics[stat].append(scoreDict[stat](truth_filtered, pred_filtered))

                print('metrics: ', metrics)
                print('filtered: ',filtered_metrics)

    return round(metrics['RMSE'][0],2), round(filtered_metrics['RMSE'][0],2)

