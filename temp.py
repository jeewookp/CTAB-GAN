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
df_train = pd.read_csv('/home/ec2-user/SageMaker/CTAB-GAN/first_data/ML_data_dev.csv')
df_train = df_train.drop(['dev_val', 'pk'], axis=1)
categorical_columns = list(df_train.columns)

synthesizer = CTABGANSynthesizer(epochs = epochs)
data_prep = DataPrep(raw_df=df_train, categorical=categorical_columns, log=[], mixed={}, integer=[],
                     type={"Classification": 'bad'})


synthesizer.fit(train_data=data_prep.df, categorical = data_prep.column_types["categorical"],
        mixed = data_prep.column_types["mixed"],type={"Classification": 'bad'})

sample = synthesizer.sample(len(df_train))
syn = data_prep.inverse_prep(sample)




# import torch.nn as nn
# import torch.nn.functional as F
# import random
#
# class Conv_Relu_Conv(torch.nn.Module):
#     def __init__(self, in_dims, hidden_dims, out_dims):
#         super(Conv_Relu_Conv,self).__init__()
#         self.fc0 = nn.Linear(in_dims,hidden_dims)
#         self.fc1 = nn.Linear(hidden_dims,out_dims)
#
#     def forward(self, input):
#         input = F.relu(self.fc0(input))
#         input = self.fc1(input)
#         return input
#
# hidden_dims = 512
# feature_dims = 10
# batch_size = 512
#
# df_train = pd.read_csv('/Users/john/data/ML_data_dev.csv')
# df_test = pd.read_csv('/Users/john/data/ML_data_val.csv')
#
# x_train = torch.from_numpy(df_train.drop(['dev_val', 'pk','bad'], axis=1).values).to(torch.float32)
# x_test = torch.from_numpy(df_test.drop(['dev_val', 'pk','bad'], axis=1).values).to(torch.float32)
#
# Y_train = torch.from_numpy(pd.get_dummies(df_train.bad).values[:,0])
# Y_test = torch.from_numpy(pd.get_dummies(df_test.bad).values[:,0])
#
#
# x_train = torch.from_numpy(sample[:,:-1]).to(torch.float32)
# Y_train = torch.from_numpy(1-sample[:,-1]).to(torch.uint8)
#
#
# classifier = Conv_Relu_Conv(x_train.shape[1],hidden_dims,2)
# optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
# criterion = nn.CrossEntropyLoss()
#
# best = 0
# for i in range(30000):
#     classifier.train()
#     samples = random.sample(range(x_train.shape[0]),batch_size)
#     x_train_unit = x_train[samples]
#     Y_train_unit = Y_train[samples]
#     output = classifier(x_train_unit)
#     loss = criterion(output,Y_train_unit)
#     classifier.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if i==0:
#         avg_loss = loss.item()
#     else:
#         avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
#     if i%1000==0:
#         classifier.eval()
#         print(i, avg_loss)
#         output = classifier(x_test)
#         output = torch.exp(output)
#         output = output[:,0]/(output[:,0]+output[:,1])
#         sorted_indices = torch.argsort(output)
#         sorted_labels = Y_test[sorted_indices]
#         n_positives = torch.cumsum(sorted_labels,dim=0)
#         n_negatives = torch.arange(1,n_positives.shape[0]+1) - n_positives
#         cum_pos_ratio = n_positives/n_positives[-1]
#         cum_neg_ratio = n_negatives/n_negatives[-1]
#         KS = torch.max(cum_pos_ratio - cum_neg_ratio)
#         print(KS.item())
#         if best<KS.item():
#             best = KS.item()
#             print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')





