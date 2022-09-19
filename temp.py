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

epochs = 10

synthesizer = CTABGANSynthesizer(epochs = epochs)
df_train = pd.read_csv('/home/ec2-user/SageMaker/CTAB-GAN/first_data/ML_data_dev.csv')
df_train = df_train.drop(['dev_val', 'pk'], axis=1)
categorical_columns = list(df_train.columns)
data_prep = DataPrep(raw_df=df_train, categorical=categorical_columns, log=[], mixed={}, integer=[],
                     type={"Classification": 'bad'})
synthesizer.fit(data_prep=data_prep, type={"Classification": 'bad'})


sample = synthesizer.sample(len(df_train))
syn = data_prep.inverse_prep(sample)
syn.to_csv('/home/ec2-user/SageMaker/CTAB-GAN/first_data/ML_data_fake.csv', index=False)

