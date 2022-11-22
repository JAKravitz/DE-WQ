"""
Copy of Jeremy Kravitz's MLP_retrieval.py, but in PyTorch
TensorFlow version: https://github.com/JAKravitz/JPL_WQ_ML/blob/main/MLP_retrieval.py

Laurel Hopkins Manella
June 23, 2022
"""

from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import os
import random
import pickle

class WQ_Data(BaseEstimator):

    def __init__(self, batch_info):
        self.sensor = batch_info['sensor']
        self.targets = batch_info['targets']
        self.meta = batch_info['meta']
        self.Xpca = batch_info['Xpca']
        self.vars = None
        self.n_in = None
        self.n_out = None
        self.Xscaler = None
        self.yscaler = None
        self.pca = None

    def clean(self, data):
        data = data.replace([np.inf, -np.inf], np.nan, inplace=False)
        data = data.dropna(axis=0, how='any', inplace=False)
        return data

    def clean_y(self, data):
        data = data.replace([-999], np.nan, inplace=False)
        data = data.dropna(axis=0, how='any', inplace=False)
        return data

    def l2norm(self, data):
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer(norm='l2')
        data = scaler.fit_transform(data)
        return data, scaler

    def save_scalers(self, xScalerFname, yScalerFname):
        if self.Xscaler is not None:
            pickle.dump(self.Xscaler, open(xScalerFname,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        if self.yscaler is not None:
            pickle.dump(self.yscaler, open(yScalerFname,'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    def load_scalers(self, xScalerFname, yScalerFname):
        if xScalerFname is not None:
            self.Xscaler = pickle.load(open(xScalerFname,'rb'))
        if yScalerFname is not None:
            self.yscaler = pickle.load(open(yScalerFname,'rb'))


    def standardScaler_X(self, data, split):
        from sklearn.preprocessing import StandardScaler
        if split == 'train':
            self.Xscaler = StandardScaler()
            data = self.Xscaler.fit_transform(data)  # only fit to training data
        else:
            data = self.Xscaler.transform(data)
        return data

    def standardScaler_y(self, data, split):
        from sklearn.preprocessing import StandardScaler
        if split == 'train':
            self.yscaler = StandardScaler()
            data = self.yscaler.fit_transform(data)  # only fit to training data
        else:
            data = self.yscaler.transform(data)
        return data


    def save_pca(self, pcaFname):
        if self.pca is not None:
            pickle.dump(self.pca, open(pcaFname,'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    def load_pca(self, pcaFname):
        if pcaFname is not None:
            self.pca = pickle.load(open(pcaFname,'rb'))


    def nPCA(self, data, n, split):
        from sklearn.decomposition import PCA
        if split == 'train':
            self.pca = PCA(n_components=n)
            self.pca.fit(data)
            npca = self.pca.transform(data)
            self.Xcomp = self.pca.components_
            self.Xvar = self.pca.explained_variance_ratio_
        else:
            npca = self.pca.transform(data)
        return npca


    def nPCA_revert(self, data):
        revert = np.dot(data, self.Xcomp)
        return revert

    def transform_inverse(self, data):
        inverse = self.yscaler.inverse_transform(data)
        return inverse

    def transform_inverse_uncertainty(self, uncert):
        if self.yscaler.scale_ is not None:
            return uncert * self.yscaler.scale_

    def get_water_types(self, clus):
        # change cluster values to OWT
        clus = (clus + 1).astype(int)

        #unique_c, counts_c = np.unique(clus, return_counts=True)  # unique clusters
        clus = clus.replace(to_replace=[2, 5, 11], value='Mild')
        clus = clus.replace(to_replace=[1, 12], value='NAP')
        clus = clus.replace(to_replace=[8, 13], value='CDOM')
        clus = clus.replace(to_replace=[7], value='Euk')
        clus = clus.replace(to_replace=[6, 9], value='Cy')
        clus = clus.replace(to_replace=[3], value='Scum')
        clus = clus.replace(to_replace=[4, 10], value='Oligo')

        clus = clus.to_numpy()
        #unique_w, counts_w = np.unique(clus, return_counts=True)  # unique water types

        return clus


    def getXY(self, data, print_stats=True):
        # get sensor data
        sensors = {'s2_60m': ['443', '490', '560', '665', '705', '740', '783', '842', '865'],
                   's2_20m': ['490', '560', '665', '705', '740', '783', '842', '865'],
                   's2_10m': ['490', '560', '665', '842'],
                   's3': '^Oa',
                   'l8': ['Aer', 'Blu', 'Grn', 'Red', 'NIR'],
                   'modis': '^RSR',
                   'meris': '^b',
                   'hico': '^H'}

        if self.sensor in ['s2_20m', 's2_10m']:
            X = data['s2'].filter(items=sensors[self.sensor])
            data = data['s2']
        elif self.sensor == 's2_60m':
            X = data['s2'].filter(regex='^[0-9]')
            data = data['s2']
        else:
            X = data[self.sensor].filter(regex='^[0-9]')
            data = data[self.sensor]

        # drop o2 bands if s3
        if self.sensor == 's3':
            X.drop(['761.25', '764.375', '767.75'], axis=1, inplace=True)

        # drop if modis
        if self.sensor == 'modis':
            X.drop(['551'], axis=1, inplace=True)

        # get meta columns
        if self.meta:
            metacols = ['cluster', ' SZA', ' OZA', ' SAA', ' OAA', ' aot550', ' astmx', ' ssa400',
                        ' ssa675', ' ssa875', ' altitude', ' adjFactor', ]
            meta = data.loc[:, metacols]
            meta[' adjFactor'] = meta[' adjFactor'].replace(to_replace=(' '), value=0)
            meta[' adjFactor'] = [float(i) for i in meta[' adjFactor']]
            X = pd.concat([X, meta], axis=1)

        # get outputs
        # t = [x for x in self.targets if x is not 'cluster']
        y = data[self.targets]
        cluster = data['cluster']

        if print_stats:
            y_min = y[self.targets[0]].min()
            y_max = y[self.targets[0]].max()
            y_mean = y[self.targets[0]].mean()
            y_sd = y[self.targets[0]].std()
            y_50th = y[self.targets[0]].quantile(q=0.5)
            y_75th = y[self.targets[0]].quantile(q=0.75)
            print(f'\n{self.targets[0]} min:', y_min)
            print(f'{self.targets[0]} max:', y_max)
            print(f'{self.targets[0]} mean:', y_mean)
            print(f'{self.targets[0]} stdDev:', y_sd)
            print(f'{self.targets[0]} 50th percentile:', y_50th)
            print(f'{self.targets[0]} 75th percentile:', y_75th, '\n')

        return X, y, cluster


    def preprocessXY(self, X, y, conf, output_path, split, plot_hist=False):
        print(f'preprocessed {split}:', X.shape)
        X = self.clean(X)  # drops rows w/ -inf or inf
        y = y.loc[X.index, :]
        y = self.clean_y(y)
        X = X.loc[y.index, :]

        Xlog = np.where(X > 0, np.log(X), X)

        Xt = self.standardScaler_X(Xlog, split)
        Xt = pd.DataFrame(Xt, columns=X.columns)

        if plot_hist:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.hist(y.values, bins=40)
            ax.set_xlabel(f'{self.targets[0]} concentration [mg/m^3]')
            ax.set_ylabel('counts')
            ax.set_title(f'Histogram of {self.targets[0]} concentration')
            title = f'Hist_{self.targets[0]}_{split}'
            if output_path is not None:
                fig.savefig(os.path.join(output_path, title + ".png"),
                    dpi=300, bbox_inches="tight")
            else:
                fig.savefig(os.path.join(conf["home_dir"], conf["output"]["out_dir"], title + ".png"),
                        dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()

        # scale/transform y
        if 'cluster' in self.targets:
            y['cluster'] = y['cluster'] + 1
        y = y + .001
        if 'cluster' in self.targets:
            y['cluster'] = round(y['cluster'])
        # ylog = np.where(y>0,np.log(y),y)
        ylog = np.log(y)

        if plot_hist:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.hist(ylog, bins=40)
            ax.set_xlabel(f'log ({self.targets[0]} concentration) [mg/m^3]')
            ax.set_ylabel('counts')
            ax.set_title(f'Histogram of logged {self.targets[0]} concentration')
            title = f'Hist_log_{self.targets[0]}_{split}'
            if output_path is not None:
                fig.savefig(os.path.join(output_path, title + ".png"),
                            dpi=300, bbox_inches="tight")
            else:
                fig.savefig(os.path.join(conf["home_dir"], conf["output"]["out_dir"], title + ".png"),
                            dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()

        yt = self.standardScaler_y(ylog, split)
        y2 = pd.DataFrame(yt, columns=y.columns)

        if plot_hist:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.hist(yt, bins=40)
            ax.set_xlabel(f'standardized log( {self.targets[0]}) [mg/m^3]')
            ax.set_ylabel('counts')
            ax.set_title(f'Histogram of standardized {self.targets[0]} concentration')
            title = f'Hist_standardized_log_{self.targets[0]}_{split}'
            if output_path is not None:
                fig.savefig(os.path.join(output_path, title + ".png"),
                            dpi=300, bbox_inches="tight")
            else:
                fig.savefig(os.path.join(conf["home_dir"], conf["output"]["out_dir"], title + ".png"),
                            dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()

        # add realistic instrument Noise
        # Signal to Noise ratio
        SNR = 200
        # draw random numbers from Gaussian
        N = np.random.randn(Xt.shape[0], Xt.shape[1])
        # calculate mean reflectance for each wavelength
        m = np.mean(Xt, 0)
        m = np.vstack([m]*Xt.shape[0])  # make m same shape as N
        # add noise to reflectances
        Xt += m / SNR * N

        # PCA for X
        if self.Xpca is not None:
            # requires transform
            Xt = self.nPCA(Xt.values, int(self.Xpca), split)
            Xt = pd.DataFrame(Xt)

        self.n_in = Xt.shape[1]
        self.n_out = y2.shape[1]
        self.vars = y2.columns.values

        print(f'{split} input:', Xt.shape)

        return Xt, y2, y


    def save_split(self, X_train, X_test, X_val, y_train, y_test, y_val, clust_train,
                  clust_test, clust_val, saved_model_dir):
        split = {'X_train': X_train,
                 'X_test': X_test,
                 'X_val': X_val,
                 'y_train': y_train,
                 'y_test': y_test,
                 'y_val': y_val,
                 'clust_train': clust_train,
                'clust_test': clust_test,
                 'clust_val': clust_val}

        splitFname = os.path.join(saved_model_dir, 'train_test_split.pkl')
        pickle.dump(split, open(splitFname,'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    def load_split(self, splitFname):
        if splitFname is not None:
            data_split = pickle.load(open(splitFname, 'rb'))
        split = []
        for key in data_split.keys():
            split.append(data_split[key])
        split = tuple(split)
        return split


    def random_train_test_val_split(self, x, y, clusters, test_ratio, val_ratio, saved_model_dir):
        x_y = x.join(y, how='inner')
        dataset_size = x_y.shape[0]
        num_test_exs = int(test_ratio * dataset_size)
        num_val_exs = int(val_ratio * dataset_size)
        all_indices = np.arange(dataset_size)
        test_indices = np.random.choice(dataset_size, size=num_test_exs, replace=False)
        train_val_indices = np.setdiff1d(all_indices, test_indices)
        val_indices = np.random.choice(train_val_indices, size=num_val_exs, replace=False)
        train_indices = np.setdiff1d(train_val_indices, val_indices)

        # ensure there isn't overlap between test and val splits
        check_overlap = np.concatenate((train_indices, test_indices, val_indices))
        u, c = np.unique(check_overlap, return_counts=True)
        dup = u[c > 1]
        if len(dup) > 0:
            raise Exception("Traint, test and val splits overlap. Redefine splits.")

        for var in y.columns:  # TODO: validate for >1 target variables
            # subset based on indices
            X = x_y.drop(var, axis=1)
            y = x_y[[var]]
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            X_val = X.iloc[val_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
            y_val = y.iloc[val_indices]

        clust = pd.DataFrame(clusters, columns=['cluster'])
        clust_train = clust.iloc[train_indices]
        clust_test = clust.iloc[test_indices]
        clust_val = clust.iloc[val_indices]

        self.save_split(X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test,
                       clust_val, saved_model_dir)

        return X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test, clust_val


    def stratified_train_test_val_split(self, x, y, clusters, test_ratio, val_ratio, saved_model_dir):
        y_var = y.columns.values[0]  # update for >1 targets
        x_y = x.join(y, how='inner')

        water_types = self.get_water_types(clusters)
        if x_y.shape[0] == len(water_types):
            x_y['water_types'] = water_types
        else:
            raise Exception("Size of XY and len(water_types) does not match.")

        dataset_size = x_y.shape[0]
        num_test_exs = round(test_ratio * dataset_size)
        num_val_exs = round(val_ratio * dataset_size)
        unique_water_types, counts = np.unique(water_types, return_counts=True)
        wt_counts = dict(zip(unique_water_types, counts))

        test_indices, val_indices = [], []
        for type in unique_water_types:
            type_indices = x_y.index[x_y['water_types'] == type].tolist()
            # calculate how many test & val exaples to take from the given water type
            percentage_of_dataset = wt_counts[type] / dataset_size
            num_test_exs_of_type = round(percentage_of_dataset * num_test_exs)
            num_val_exs_of_type = round(percentage_of_dataset * num_val_exs)

            # randomly sample examples from the given water type
            test_val_indices = np.random.choice(type_indices, size=num_test_exs_of_type + num_val_exs_of_type, replace=False)
            test_indices.extend(test_val_indices[0:num_test_exs_of_type])
            val_indices.extend(test_val_indices[num_test_exs_of_type:])

        test_val = test_indices + val_indices
        #train_indices = np.setdiff1d(np.arange(dataset_size), test_indices + val_indices).tolist()
        train_indices = np.setdiff1d(np.arange(dataset_size), test_val).tolist()

        # ensure there isn't overlap between test and val splits
        all_indices = train_indices + test_indices + val_indices
        unique_indices, counts = np.unique(all_indices, return_counts=True)
        dup = unique_indices[counts > 1]
        if len(dup) > 0:
            raise Exception("Train, test, and val splits overlap. Redefine splits.")

        # subset based on indices
        X = x_y.drop([y_var, 'water_types'], axis=1)
        y = x_y[[y_var]]
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        X_val = X.iloc[val_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        y_val = y.iloc[val_indices]

        clust = pd.DataFrame(clusters, columns=['cluster'])
        clust_train = clust.iloc[train_indices]
        clust_test = clust.iloc[test_indices]
        clust_val = clust.iloc[val_indices]

        self.save_split(X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test,
                       clust_val, saved_model_dir)

        return X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test, clust_val


    def water_type_train_test_val_split(self, x, y, clusters, val_ratio, test_water_type, saved_model_dir):
        print('Test water type: ', test_water_type)

        y_var = y.columns.values[0]  # update for >1 targets
        x_y = x.join(y, how='inner')
        dataset_size = x_y.shape[0]

        water_types = self.get_water_types(clusters)
        if x_y.shape[0] == len(water_types):
            x_y['water_types'] = water_types
        else:
            raise Exception("Size of XY and len(water_types) does not match.")

        unique_water_types, counts = np.unique(water_types, return_counts=True)

        # assign test/val indices based on split_assignment
        water_types_remaining = list(unique_water_types)  # update as water types have been assigned
        # 1. water type is specified
        if test_water_type != 'random':
            test_indices = x_y.index[x_y['water_types'] == test_water_type].tolist()
            water_types_remaining.remove(test_water_type)

        # 2. randomly select water type
        else:
            split_type = random.choice(water_types_remaining)
            test_indices = x_y.index[x_y['water_types'] == split_type].tolist()
            water_types_remaining.remove(split_type)

        train_val_indices = np.setdiff1d(np.arange(dataset_size), test_indices).tolist()


        # split test indices
        X = x_y.drop([y_var, 'water_types'], axis=1)
        y = x_y[[y_var]]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]
        X_train_val = X.iloc[train_val_indices]
        X_train_val = X_train_val.reset_index()
        y_train_val = y.iloc[train_val_indices]
        y_train_val = y_train_val.reset_index()
        y_train_val = y_train_val.drop(['index'], axis=1)

        clust = pd.DataFrame(clusters, columns=['cluster'])
        clust_test = clust.iloc[test_indices]
        clust_train_val = clust.iloc[train_val_indices]
        #clust_train_val = clust_train_val.reset_index()

        # get val indices
        strat_split = self.stratified_train_test_val_split(X_train_val, y_train_val, clust_train_val,
                                                           0, val_ratio)

        X_train, _, X_val, y_train, _, y_val, clust_train, _clust_test, clust_val = strat_split
        X_train = X_train.drop(['index'], axis=1)
        X_val = X_val.drop(['index'], axis=1)

        self.save_split(X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test,
                       clust_val, saved_model_dir)

        return X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test, clust_val


    def binned_train_test_val_split(self, x, y, clusters, test_ratio, val_ratio, saved_model_dir):
        x_y = x.join(y, how='inner')
        for var in y.columns:  # TODO: validate for >1 target variables
            dataset_size = x_y.shape[0]
            num_test_exs = int(test_ratio * dataset_size)
            num_val_exs = int(val_ratio * dataset_size)
            num_binned_exs = num_test_exs + num_val_exs

            # define bounds for test/val indices
            min_start_index = round(0.1 * dataset_size)
            max_start_index = dataset_size - num_binned_exs - round(0.1 * dataset_size)
            start_index = np.random.randint(low=min_start_index, high=max_start_index)
            test_indices = list(range(start_index, start_index+num_test_exs))
            val_indices = list(range(start_index+num_test_exs, start_index+num_test_exs+num_val_exs))
            train_indices = np.arange(0, dataset_size)
            train_indices = np.setdiff1d(train_indices, test_indices + val_indices)

        # subset based on indices
        x_y = x_y.sort_values(by=[var])
        X = x_y.drop(var, axis=1)
        y = x_y[[var]]
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        X_val = X.iloc[val_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        y_val = y.iloc[val_indices]

        clust = pd.DataFrame(clusters, columns=['cluster'])
        clust_train = clust.iloc[train_indices]
        clust_test = clust.iloc[test_indices]
        clust_val = clust.iloc[val_indices]

        self.save_split(X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test,
                       clust_val, saved_model_dir)

        return X_train, X_test, X_val, y_train, y_test, y_val, clust_train, clust_test, clust_val

    def prep_results(self, y):
        results = {}
        for var in y.columns:
            if var in ['cluster']:
                continue
            results[var] = {'cv': {'ytest': [],
                                   'yhat': [],
                                   'uncertainty': [],
                                   'R2': [],
                                   'RMSE': [],
                                   'RMSELE': [],
                                   'Bias': [],
                                   'MAPE': [],
                                   'PICP': [],
                                   'MPIW': [],
                                   'rRMSE': []},
                            'final': {'ytest': [],
                                      'yhat': [],
                                      'uncertainty': [],
                                      'R2': [],
                                      'RMSE': [],
                                      'RMSELE': [],
                                      'Bias': [],
                                      'MAPE': [],
                                      'PICP': [],
                                      'MPIW': [],
                                      'rRMSE': []},
                            'owt': {'Mild': [],
                                    'NAP': [],
                                    'CDOM': [],
                                    'Euk': [],
                                    'Cy': [],
                                    'Scum': [],
                                    'Oligo': []}
                            }

            clusters = ['Mild', 'NAP', 'CDOM', 'Euk', 'Cy', 'Scum', 'Oligo', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
            for k in clusters:
                results[var]['owt'][k] = {'ytest': [],
                                          'yhat': [],
                                          'uncertainty': [],
                                          'R2': [],
                                          'RMSE': [],
                                          'RMSELE': [],
                                          'Bias': [],
                                          'MAPE': [],
                                          'PICP': [],
                                          'MPIW': [], 
                                          'rRMSE': []}
            results['fit_time'] = []
            results['pred_time'] = []
            results['train_loss'] = []
            results['val_loss'] = []
        return results

