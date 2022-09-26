import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import random
import torch.nn as nn
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import pickle

with open('/Users/john/project/CTAB-GAN/model/synthesizer.pickle','rb') as read_file:
    synthesizer = pickle.load(read_file)

class Conv_Relu_Conv(torch.nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(Conv_Relu_Conv,self).__init__()
        self.fc0 = nn.Linear(in_dims,hidden_dims)
        self.fc1 = nn.Linear(hidden_dims,out_dims)

    def forward(self, input):
        input = F.relu(self.fc0(input))
        input = self.fc1(input)
        return input

def evaluate_function(x,Y,classifier):
    with torch.no_grad():
        output = classifier(x)
    output = output[:, 0] - output[:, 1]
    output = torch.sigmoid(output)
    sorted_indices = torch.argsort(output)
    sorted_labels = Y[sorted_indices]
    n_positives = torch.cumsum(sorted_labels, dim=0)
    n_negatives = torch.arange(1, n_positives.shape[0] + 1) - n_positives
    cum_pos_ratio = n_positives / n_positives[-1]
    cum_neg_ratio = n_negatives / n_negatives[-1]
    KS = torch.max(cum_pos_ratio - cum_neg_ratio)
    return KS.item()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        # outlier one hot encoding
        outlier = list(item.keys())[0]
        key = f'{col}_{outlier}'
        df2 = pd.DataFrame({key:np.zeros(len(df))})
        df = pd.concat([df, df2], axis=1)
        df.loc[df[col] == outlier, [key]] = 1
        # replace outlier to mean/mode
        df[col] = df[col].replace(outlier, mean_mode_dict[col]['val'])

scaler = MinMaxScaler()
x0 = df[df['val_gb_new'] == 0]
x1 = df[df['val_gb_new'] == 1]
x2 = df[df['val_gb_new'] == 2]

val_indices = np.load('first_data/x0_val.npy')
train_indices = sorted(list(set(x0.index)-set(val_indices)))
x0_train = x0.loc[train_indices]
x0_val = x0.loc[val_indices]

val_indices = np.load('first_data/x1_val.npy')
train_indices = sorted(list(set(x1.index)-set(val_indices)))
x1_train = x1.loc[train_indices]
x1_val = x1.loc[val_indices]

y0_train = x0_train['target_6m']
x0_train_temp = x0_train.drop(['target_6m', 'val_gb_new'], axis=1)
cat_scaler = scaler.fit(x0_train_temp)
x0_train = cat_scaler.transform(x0_train_temp)
x0_train = torch.from_numpy(x0_train).to(torch.float32).to(device)
y0_train = torch.from_numpy(pd.get_dummies(y0_train).values[:,0]).to(device)

y0_val = x0_val['target_6m']
x0_val_temp = x0_val.drop(['target_6m', 'val_gb_new'], axis=1)
x0_val = cat_scaler.transform(x0_val_temp)
x0_val = torch.from_numpy(x0_val).to(torch.float32).to(device)
y0_val = torch.from_numpy(pd.get_dummies(y0_val).values[:,0]).to(device)

y1_train = x1_train['target_6m']
x1_train_temp = x1_train.drop(['target_6m', 'val_gb_new'], axis=1)
x1_train = cat_scaler.transform(x1_train_temp)
x1_train = torch.from_numpy(x1_train).to(torch.float32).to(device)
y1_train = torch.from_numpy(pd.get_dummies(y1_train).values[:,0]).to(device)

y1_val = x1_val['target_6m']
x1_val_temp = x1_val.drop(['target_6m', 'val_gb_new'], axis=1)
x1_val = cat_scaler.transform(x1_val_temp)
x1_val = torch.from_numpy(x1_val).to(torch.float32).to(device)
y1_val = torch.from_numpy(pd.get_dummies(y1_val).values[:,0]).to(device)

y2 = x2['target_6m']
x2_temp = x2.drop(['target_6m', 'val_gb_new'], axis=1)
x2 = cat_scaler.transform(x2_temp)
x2 = torch.from_numpy(x2).to(torch.float32).to(device)
y2 = torch.from_numpy(pd.get_dummies(y2).values[:,0]).to(device)

df_fake = pd.read_csv("first_data/seg200_fake.csv")
y0_fake = df_fake['target_6m']
x0_fake_temp = df_fake.drop(['target_6m'], axis=1)
x0_fake = cat_scaler.transform(x0_fake_temp)
x0_fake = torch.from_numpy(x0_fake).to(torch.float32).to(device)
y0_fake = torch.from_numpy(pd.get_dummies(y0_fake).values[:,0]).to(device)



classifier = Conv_Relu_Conv(x0_train.shape[1],1024,2).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

batch_size = 500
best = 0
for i in range(30000):
    classifier.train()
    samples = random.sample(range(x0_train.shape[0]),batch_size)
    x_train_unit = x0_train[samples]
    Y_train_unit = y0_train[samples]
    output = classifier(x_train_unit)
    loss = criterion(output,Y_train_unit)

    classifier.zero_grad()
    loss.backward()
    optimizer.step()
    if i==0:
        avg_loss = loss.item()
    else:
        avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
    if i%1000==0:
        classifier.eval()
        print(i, avg_loss)

        KS0_val = evaluate_function(x0_val,y0_val,classifier)
        KS1_val = evaluate_function(x1_val,y1_val,classifier)
        KS2 = evaluate_function(x2,y2,classifier)

        print(KS0_val, KS1_val, KS2)
        if best<KS0_val:
            classifier_save = copy.deepcopy(classifier)
            best = KS0_val
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')





classifier = copy.deepcopy(classifier_save)
classifier.eval()
precision_matrices = {}
origin_params = {}
for n, p in copy.deepcopy({n: p for n, p in classifier.named_parameters()}).items():
    precision_matrices[n] = 0
    origin_params[n] = p.data
for i in range(x0_train.shape[0]):
    if i%1000==0:
        print(i)
    x_train_unit = x0_train[i:i+1]
    Y_train_unit = y0_train[i:i+1]
    classifier.zero_grad()
    output = classifier(x_train_unit)
    loss = criterion(output, Y_train_unit)
    loss.backward()
    for n, p in classifier.named_parameters():
        precision_matrices[n] += p.grad.data ** 2
for n, p in classifier.named_parameters():
    precision_matrices[n] /= x0_train.shape[0]

with torch.no_grad():
    y0_fake_inferenced = classifier(x0_fake)
y0_fake_inferenced = y0_fake_inferenced[:, 1] - y0_fake_inferenced[:, 0]
y0_fake_inferenced = torch.sigmoid(y0_fake_inferenced)
x0_fake_1_train = torch.cat([x0_fake,x1_train],dim=0)
y0_fake_1_train = torch.cat([y0_fake_inferenced,y1_train],dim=0)

importance = 1
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
best = 0
for i in range(30000):
    classifier.train()
    samples = random.sample(range(x0_fake_1_train.shape[0]),batch_size)
    x_train_unit = x0_fake_1_train[samples]
    Y_train_unit = y0_fake_1_train[samples]
    output = classifier(x_train_unit)
    output = output[:, 1] - output[:, 0]
    output = torch.sigmoid(output)
    output = 1e-6 + (1-2e-6)*output
    loss = -torch.mean(Y_train_unit*torch.log(output)+(1-Y_train_unit)*torch.log(1-output))

    loss_ewc = 0
    for n, p in classifier.named_parameters():
        _loss = precision_matrices[n] * (p - origin_params[n]) ** 2
        loss_ewc += _loss.sum()

    classifier.zero_grad()
    (loss+importance * loss_ewc).backward()
    optimizer.step()
    if i==0:
        avg_loss = loss.item()
        avg_loss_ewc = loss_ewc.item()
    else:
        avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
        avg_loss_ewc = 0.999 * avg_loss_ewc + 0.001 * loss_ewc.item()
    if i%1000==0:
        classifier.eval()
        print(i, avg_loss, avg_loss_ewc)

        KS0_val = evaluate_function(x0_val,y0_val,classifier)
        KS1_val = evaluate_function(x1_val,y1_val,classifier)
        KS2 = evaluate_function(x2,y2,classifier)

        print(KS0_val, KS1_val, KS2)
        if best<KS1_val:
            best = KS1_val
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')






classifier = copy.deepcopy(classifier_save)
classifier.eval()
with torch.no_grad():
    y0_fake_inferenced = classifier(x0_fake)

y0_fake_inferenced = y0_fake_inferenced[:, 1] - y0_fake_inferenced[:, 0]
y0_fake_inferenced = torch.sigmoid(y0_fake_inferenced)

x0_fake_1_train = torch.cat([x0_fake,x1_train],dim=0)
y0_fake_1_train = torch.cat([y0_fake_inferenced,y1_train],dim=0)

classifier = copy.deepcopy(classifier_save)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
best = 0
for i in range(30000):
    classifier.train()
    samples = random.sample(range(x0_fake_1_train.shape[0]),batch_size)
    x_train_unit = x0_fake_1_train[samples]
    Y_train_unit = y0_fake_1_train[samples]
    output = classifier(x_train_unit)
    output = output[:, 1] - output[:, 0]
    output = torch.sigmoid(output)
    output = 1e-6 + (1-2e-6)*output
    loss = -torch.mean(Y_train_unit*torch.log(output)+(1-Y_train_unit)*torch.log(1-output))
    classifier.zero_grad()
    loss.backward()
    optimizer.step()
    if i==0:
        avg_loss = loss.item()
    else:
        avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
    if i%1000==0:
        classifier.eval()
        print(i, avg_loss)

        KS0_val = evaluate_function(x0_val,y0_val,classifier)
        KS1_val = evaluate_function(x1_val,y1_val,classifier)
        KS2 = evaluate_function(x2,y2,classifier)

        print(KS0_val, KS1_val, KS2)
        if best<KS1_val:
            best = KS1_val
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')














classifier = copy.deepcopy(classifier_save)
precision_matrices = {}
origin_params = {}
for n, p in copy.deepcopy({n: p for n, p in classifier.named_parameters()}).items():
    precision_matrices[n] = 0
    origin_params[n] = p.data
classifier.eval()
for i in range(x0_train.shape[0]):
    if i%1000==0:
        print(i)
    x_train_unit = x0_train[i:i+1]
    Y_train_unit = y0_train[i:i+1]
    classifier.zero_grad()
    output = classifier(x_train_unit)
    loss = criterion(output, Y_train_unit)
    loss.backward()
    for n, p in classifier.named_parameters():
        precision_matrices[n] += p.grad.data ** 2
for n, p in classifier.named_parameters():
    precision_matrices[n] /= x0_train.shape[0]

importance = 100
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
best = 0
for i in range(30000):
    classifier.train()
    samples = random.sample(range(x1_train.shape[0]),batch_size)
    x_train_unit = x1_train[samples]
    Y_train_unit = y1_train[samples]
    output = classifier(x_train_unit)
    loss = criterion(output,Y_train_unit)

    loss_ewc = 0
    for n, p in classifier.named_parameters():
        _loss = precision_matrices[n] * (p - origin_params[n]) ** 2
        loss_ewc += _loss.sum()

    classifier.zero_grad()
    (loss+importance * loss_ewc).backward()
    optimizer.step()
    if i==0:
        avg_loss = loss.item()
        avg_loss_ewc = loss_ewc.item()
    else:
        avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
        avg_loss_ewc = 0.999 * avg_loss_ewc + 0.001 * loss_ewc.item()
    if i%1000==0:
        classifier.eval()
        print(i, avg_loss, avg_loss_ewc)

        x_train_unit = x0_train
        Y_train_unit = y0_train
        with torch.no_grad():
            output = classifier(x_train_unit)
        print(criterion(output, Y_train_unit).item())

        KS0_val = evaluate_function(x0_val,y0_val,classifier)
        KS1_val = evaluate_function(x1_val,y1_val,classifier)
        KS2 = evaluate_function(x2,y2,classifier)

        print(KS0_val, KS1_val, KS2)
        if best<KS1_val:
            best = KS1_val
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')




classifier = copy.deepcopy(classifier_save)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
best = 0
for i in range(30000):
    classifier.train()
    samples = random.sample(range(x1_train.shape[0]),batch_size)
    x_train_unit = x1_train[samples]
    Y_train_unit = y1_train[samples]
    output = classifier(x_train_unit)
    loss = criterion(output,Y_train_unit)
    classifier.zero_grad()
    loss.backward()
    optimizer.step()
    if i==0:
        avg_loss = loss.item()
    else:
        avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
    if i%1000==0:
        classifier.eval()
        print(i, avg_loss)

        KS0_val = evaluate_function(x0_val,y0_val,classifier)
        KS1_val = evaluate_function(x1_val,y1_val,classifier)
        KS2 = evaluate_function(x2,y2,classifier)

        print(KS0_val, KS1_val, KS2)
        if best<KS1_val:
            best = KS1_val
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


x0_fake_1_train = torch.cat([x0_fake,x1_train],dim=0)
y0_fake_1_train = torch.cat([y0_fake,y1_train],dim=0)

classifier = Conv_Relu_Conv(x0_train.shape[1],1024,2).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

best = 0
for i in range(30000):
    classifier.train()
    samples = random.sample(range(x0_fake_1_train.shape[0]),batch_size)
    x_train_unit = x0_fake_1_train[samples]
    Y_train_unit = y0_fake_1_train[samples]
    output = classifier(x_train_unit)
    loss = criterion(output,Y_train_unit)
    classifier.zero_grad()
    loss.backward()
    optimizer.step()
    if i==0:
        avg_loss = loss.item()
    else:
        avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
    if i%1000==0:
        classifier.eval()
        print(i, avg_loss)

        KS0_val = evaluate_function(x0_val,y0_val,classifier)
        KS1_val = evaluate_function(x1_val,y1_val,classifier)
        KS2 = evaluate_function(x2,y2,classifier)

        print(KS0_val, KS1_val, KS2)
        if best<KS0_val:
            best = KS0_val
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')






