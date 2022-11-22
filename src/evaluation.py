import numpy as np
import pandas as pd
import scipy.stats as st
import src.scorers as sc
import os
import matplotlib.pyplot as plt
from src.viz_utils import examine_outliers


def evaluate(sensor_data, y_hat, uncertainty, y_test, results, q, log):
    if log:  # evaluate the log of the results
        scoreDict = {'R2': sc.r2,
                     'RMSE': sc.rmse,
                     'RMSELE': sc.rmsele,
                     'Bias': sc.bias,
                     'MAPE': sc.mape,
                     'rRMSE': sc.rrmse, }
    else:  # convert back to original units
        scoreDict = {'R2': sc.r2,
                     'RMSE': sc.log_rmse,
                     'RMSELE': sc.rmsele,
                     'Bias': sc.log_bias,
                     'MAPE': sc.log_mape,
                     'rRMSE': sc.log_rrmse, }

    # revert back to un-transformed data
    y_hat = sensor_data.transform_inverse(y_hat)
    y_test = sensor_data.transform_inverse(y_test)
    uncertainty = sensor_data.transform_inverse_uncertainty(uncertainty)
    # if !log:
        #y_hat = np.exp(y_hat)
        #y_test = np.exp(y_test)
        #uncertainty = np.exp(uncertainty)
    y_hat = pd.DataFrame(y_hat, columns=sensor_data.vars)
    y_test = pd.DataFrame(y_test, columns=sensor_data.vars)
    uncertainty = pd.DataFrame(uncertainty, columns=sensor_data.vars)

    for band in sensor_data.vars:
        if band in ['cluster']:
            continue

        y_t = y_test.loc[:, band].astype(float)
        y_h = y_hat.loc[:, band].astype(float)
        unc = uncertainty.loc[:, band].astype(float)

        for stat in scoreDict:
            results[band][q][stat].append(scoreDict[stat](y_t, y_h))

        results[band][q]['uncertainty'].append(unc) 
        results[band][q]['ytest'].append(y_t)
        results[band][q]['yhat'].append(y_h)
    return results


def owt_evaluate(sensor_data, cluster, y_hat, uncertainty, y_test, results, log):
    if log:  # evaluate the log of the results
        scoreDict = {'R2': sc.r2,
                     'RMSE': sc.rmse,
                     'RMSELE': sc.rmsele,
                     'Bias': sc.bias,
                     'MAPE': sc.mape,
                     'rRMSE': sc.rrmse, }
    else:  # convert back to original units
        scoreDict = {'R2': sc.r2,
                     'RMSE': sc.log_rmse,
                     'RMSELE': sc.rmsele,
                     'Bias': sc.log_bias,
                     'MAPE': sc.log_mape,
                     'rRMSE': sc.log_rrmse, }

    # revert back to un-transformed data, add cluster column
    clust = cluster['cluster']
    y_hat = sensor_data.transform_inverse(y_hat)
    y_test = sensor_data.transform_inverse(y_test)
    uncertainty = sensor_data.transform_inverse_uncertainty(uncertainty)
    # if !log:
        # y_hat = np.exp(y_hat)
        # y_test = np.exp(y_test)
        # uncertainty = np.exp(uncertainty)
    y_hat = pd.DataFrame(y_hat, columns=sensor_data.vars)
    y_test = pd.DataFrame(y_test, columns=sensor_data.vars)
    uncertainty = pd.DataFrame(uncertainty, columns=sensor_data.vars)

    clus = (clust+1).astype(int)


    #Commenting this out for now - currently only using water types
    #     Not underlying clusters
    """
    clus2 = clus.to_numpy()
    y_hat['cluster'] = clus2
    y_test['cluster'] = clus2

    # evaluate by cluster
    grouped = y_test.groupby('cluster')
    for c, testgroup in grouped:
        hatgroup = y_hat.loc[testgroup.index]
        uncgroup = uncertainty.loc[testgroup.index]

        for band in sensor_data.vars:
            if band in ['cluster']:
                continue

            y_t = testgroup.loc[:, band].astype(float)
            y_h = hatgroup.loc[:, band].astype(float)
            unc = uncgroup.loc[:, band].astype(float)
           
            for stat in scoreDict:
                results[band]['owt'][str(c)][stat].append(scoreDict[stat](y_t, y_h))

            results[band]['owt'][str(c)]['uncertainty'].append(unc)
            results[band]['owt'][str(c)]['ytest'].append(y_t)
            results[band]['owt'][str(c)]['yhat'].append(y_h)

    """
 
    # change cluster values to OWT
    clus = clus.replace(to_replace=[2, 5, 11], value='Mild')
    clus = clus.replace(to_replace=[1, 12], value='NAP')
    clus = clus.replace(to_replace=[8, 13], value='CDOM')
    clus = clus.replace(to_replace=[7], value='Euk')
    clus = clus.replace(to_replace=[6, 9], value='Cy')
    clus = clus.replace(to_replace=[3], value='Scum')
    clus = clus.replace(to_replace=[4, 10], value='Oligo') 

    clus = clus.to_numpy()
    y_hat['cluster'] = clus
    y_test['cluster'] = clus
 
    # evaluate by cluster
    grouped = y_test.groupby('cluster')
    for c, testgroup in grouped:
        hatgroup = y_hat.loc[testgroup.index]
        uncgroup = uncertainty.loc[testgroup.index]

        for band in sensor_data.vars:
            if band in ['cluster']:
                continue

            y_t = testgroup.loc[:, band].astype(float)
            y_h = hatgroup.loc[:, band].astype(float)
            unc = uncgroup.loc[:, band].astype(float)

            for stat in scoreDict:
                results[band]['owt'][str(c)][stat].append(scoreDict[stat](y_t, y_h))

            results[band]['owt'][str(c)]['uncertainty'].append(unc)
            results[band]['owt'][str(c)]['ytest'].append(y_t)
            results[band]['owt'][str(c)]['yhat'].append(y_h)

    return results


def evaluate_pis(y_true, y_pred, uncertainty, results, targets, q, interval):
    """
    evaluate prediction intervals (calculated from uncertainties)
    calculates (1) the prediction interval coverage probability (PICP) and (2) mean
    prediction interval width (MPIW or bandwidth)
    https://machinelearningmastery.com/prediction-intervals-for-machine-learning/
    """
    # becuase python probabilities are left-tailed by default, adjust for correct z-score
    # https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa
    tail_adj = (1.0 - interval) / 2
    z_score = st.norm.ppf(interval + tail_adj)

    # calculate prediction intervals
    y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
    pi_lower = y_pred - z_score * uncertainty
    pi_upper = y_pred + z_score * uncertainty

    # count how many predictions fall within their associated interval
    count = 0
    for i in range(len(y_true)):
        if y_pred[i] < pi_upper[i] and y_pred[i] > pi_lower[i]:
            count += 1

    picp = count / len(y_true)
    results[targets[0]][q]['PICP'] = picp

    # calcuate MPIW
    widths = pi_upper - pi_lower
    mpiw = np.mean(widths)
    results[targets[0]][q]['MPIW'] = mpiw

    return results


def evaluate_uncertainty(outlier_df, tgt, output_path):
    scoreDict = {'R2': sc.r2,
                 'RMSE': sc.rmse,
                 'RMSELE': sc.rmsele,
                 'Bias': sc.bias,
                 'MAPE': sc.mape,
                 'rRMSE': sc.rrmse, }

    filtered_metrics = {}

    outliers = outlier_df[outlier_df['type'] == 'outlier']
    test = outlier_df[outlier_df['type'] == 'test']
    num_outliers = outliers.shape[0]
    num_test = test.shape[0]

    #make hist of uncertainties
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.hist(outliers[['uncertainty']].values, label='outlier', alpha=0.4, color="orange", bins=40)
    ax.hist(test[['uncertainty']].values, label='test', alpha=0.4, color="blue", bins=40)
    ax.set_xlabel(f"standard deviation (mg/m^3)")
    ax.set_ylabel("counts")
    ax.set_title(f"Histogram of {tgt} uncertanties")
    ax.legend()
    fig.savefig(os.path.join(output_path, 'HistogramOfUncertainties.png'), dpi=300, bbox_inches="tight")

    percent_to_remove_list = [0, 0.05, 0.1, 0.15, 0.20]
    for percent_to_remove in percent_to_remove_list:
        percent_to_keep = 1.0 - percent_to_remove
        cutoff_val = np.quantile(outlier_df['uncertainty'].values, percent_to_keep)
        df_filtered = outlier_df.loc[(outlier_df['uncertainty'] <= cutoff_val)]
        truth_filtered = df_filtered[['y_test']].values.T[0]
        pred_filtered = df_filtered[['y_pred']].values.T[0]

        metrics = {'R2': [],
                   'RMSE': [],
                   'RMSELE': [],
                   'Bias': [],
                   'MAPE': [],
                   'rRMSE': []
                   }

        for stat in scoreDict:
            metrics[stat].append(scoreDict[stat](truth_filtered, pred_filtered))

        # calculate percentage of outliers removed
        remaining_outliers = df_filtered[df_filtered['type'] == 'outlier'].shape[0] / num_outliers
        remaining_test = df_filtered[df_filtered['type'] == 'test'].shape[0] / num_test
        metrics['percent_outliers'] = round(1 - remaining_outliers, 2)
        metrics['percent_test'] = round(1 - remaining_test, 2)
        filtered_metrics[percent_to_remove] = metrics

    for key in filtered_metrics.keys():
        print(f"{key*100}% of data removed - RMSE: {round(filtered_metrics[key]['RMSE'][0],2)}")
        print(f"\tremoved: {filtered_metrics[key]['percent_outliers']*100}% of outliers")
        print(f"\tremoved: {filtered_metrics[key]['percent_test']*100}% of test")

    return filtered_metrics