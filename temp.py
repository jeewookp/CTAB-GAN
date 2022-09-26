import pandas as pd
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer, Sampler, Condvec, determine_layers_gen, \
    determine_layers_disc, Generator, Discriminator, Classifier, get_st_ed, weights_init, apply_activate, cond_loss
from model.synthesizer.transformer import DataTransformer, ImageTransformer
from sklearn.mixture import BayesianGaussianMixture
from torch.optim import Adam
from tqdm import tqdm
import torch
import numpy as np

epochs = 150

synthesizer = CTABGANSynthesizer(epochs = epochs)

df = pd.read_csv('first_data/hf_model_v2_seg200_backscofing_john.csv')

outliers = [-999, -99999999, -99999900, -9999999, 9999999, 999999991, 999999992, 9999995, 999999900, 999999999, -888, -88888888, -88888800, -8888888, 8888888, 888888881, 888888882, 8888885, 888888800, 888888888]
outlier_dict = {}
mean_mode_dict = {}
for col in df.columns:
    each_df = df[col]
    # outlier check
    for out_val in outliers:
        outlier_df = df[df[col] == out_val]
        if len(outlier_df) > 0:
            if outlier_dict.get(col) is not None:
                outlier_dict[col].append({out_val:outlier_df.index})
            else:
                outlier_dict[col] = [{out_val:outlier_df.index}]
    if outlier_dict.get(col) is None:
        continue
    # distinct, mode, mean check
    outlier_indexes = []
    dict_val = outlier_dict[col]
    for item in dict_val:
        for key, val in item.items():
            outlier_indexes.extend(val)
    non_outlier_df = each_df.drop(outlier_indexes)
    n_unique = each_df.nunique()
    if n_unique < 10:
        key = 'mode'
        val = non_outlier_df.mode().values[0]
    else:
        key = 'mean'
        val = non_outlier_df.mean()
    mean_mode_dict[col] = {'key': key, 'val': val}
    for item in dict_val:
        outlier = list(item.keys())[0]
        key = f'{col}_{outlier}'
        df2 = pd.DataFrame({key:np.zeros(len(df))})
        df = pd.concat([df, df2], axis=1)
        df.loc[df[col] == outlier, [key]] = 1
        df[col] = df[col].replace(outlier, mean_mode_dict[col]['val'])

x0 = df[df['val_gb_new'] == 0]
x1 = df[df['val_gb_new'] == 1]
x2 = df[df['val_gb_new'] == 2]

val_indices = np.load('first_data/x0_val.npy')
train_indices = sorted(list(set(x0.index)-set(val_indices)))
x0_train = x0.loc[train_indices]
x0_train = x0_train.drop(['val_gb_new'], axis=1)


categorical = []
integer = []
for col in x0_train.columns:
    n_unique = x0_train[col].nunique()
    if n_unique < 10:
        categorical.append(col)
    else:
        integer.append(col)

data_prep = DataPrep(raw_df=x0_train, categorical=categorical, log=[], mixed={}, integer=integer,
                     type={"Classification": 'target_6m'})
synthesizer.fit(data_prep=data_prep, type={"Classification": 'target_6m'})


sample = synthesizer.sample(10*len(x0_train))
syn = data_prep.inverse_prep(sample)
syn.to_csv('first_data/seg200_fake.csv', index=False)

