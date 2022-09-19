import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

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
    output = classifier(x)
    output = torch.exp(output)
    output = output[:, 0] / (output[:, 0] + output[:, 1])
    sorted_indices = torch.argsort(output)
    sorted_labels = Y[sorted_indices]
    n_positives = torch.cumsum(sorted_labels, dim=0)
    n_negatives = torch.arange(1, n_positives.shape[0] + 1) - n_positives
    cum_pos_ratio = n_positives / n_positives[-1]
    cum_neg_ratio = n_negatives / n_negatives[-1]
    KS = torch.max(cum_pos_ratio - cum_neg_ratio)
    return KS.item()


batch_size = 500

df_train = pd.read_csv('/Users/john/project/CTAB-GAN/first_data/ML_data_dev.csv')
x_train = torch.from_numpy(df_train.drop(['dev_val', 'pk','bad'], axis=1).values).to(torch.float32)
Y_train = torch.from_numpy(pd.get_dummies(df_train.bad).values[:,0])

df_val = pd.read_csv('/Users/john/project/CTAB-GAN/first_data/ML_data_val.csv')
x_val = torch.from_numpy(df_val.drop(['dev_val', 'pk','bad'], axis=1).values).to(torch.float32)
Y_val = torch.from_numpy(pd.get_dummies(df_val.bad).values[:,0])

val2_selected_indices = np.load('/Users/john/project/CTAB-GAN/first_data/val2_list.npy')
val1_selected_indices = list(set(range(len(x_val)))-set(val2_selected_indices))
x_val1 = x_val[val1_selected_indices]
Y_val1 = Y_val[val1_selected_indices]
x_val2 = x_val[val2_selected_indices]
Y_val2 = Y_val[val2_selected_indices]

df_fake = pd.read_csv('/Users/john/project/CTAB-GAN/first_data/ML_data_fake.csv')
x_fake = torch.from_numpy(df_fake.drop(['bad'], axis=1).values).to(torch.float32)
Y_fake = torch.from_numpy(pd.get_dummies(df_fake.bad).values[:,0])


classifier = Conv_Relu_Conv(x_train.shape[1],512,2)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

best = 0
for i in range(30000):
    classifier.train()
    samples = random.sample(range(x_train.shape[0]),batch_size)
    x_train_unit = x_train[samples]
    Y_train_unit = Y_train[samples]
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

        KS_train = evaluate_function(x_train,Y_train,classifier)
        KS_val1 = evaluate_function(x_val1,Y_val1,classifier)
        KS_val2 = evaluate_function(x_val2,Y_val2,classifier)

        print(KS_train, KS_val1, KS_val2)
        if best<KS_val1:
            classifier_save = copy.deepcopy(classifier)
            best = KS_val1
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

classifier = copy.deepcopy(classifier_save)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
best = 0
for i in range(3000):
    classifier.train()
    samples = random.sample(range(x_val1.shape[0]),batch_size)
    x_train_unit = x_val1[samples]
    Y_train_unit = Y_val1[samples]
    output = classifier(x_train_unit)
    loss = criterion(output,Y_train_unit)
    classifier.zero_grad()
    loss.backward()
    optimizer.step()
    if i==0:
        avg_loss = loss.item()
    else:
        avg_loss = 0.999 * avg_loss + 0.001 * loss.item()
    if i%100==0:
        classifier.eval()
        print(i, avg_loss)

        KS_train = evaluate_function(x_train,Y_train,classifier)
        KS_val1 = evaluate_function(x_val1,Y_val1,classifier)
        KS_val2 = evaluate_function(x_val2,Y_val2,classifier)

        print(KS_train, KS_val1, KS_val2)
        if best<KS_val2:
            best = KS_val2
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


x_fake_val1 = torch.cat([x_fake,x_val1],dim=0)
Y_fake_val1 = torch.cat([Y_fake,Y_val1],dim=0)
classifier = Conv_Relu_Conv(x_train.shape[1],512,2)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
best = 0
for i in range(30000):
    classifier.train()
    samples = random.sample(range(x_fake_val1.shape[0]),batch_size)
    x_train_unit = x_fake_val1[samples]
    Y_train_unit = Y_fake_val1[samples]
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

        KS_train = evaluate_function(x_train,Y_train,classifier)
        KS_val1 = evaluate_function(x_val1,Y_val1,classifier)
        KS_val2 = evaluate_function(x_val2,Y_val2,classifier)

        print(KS_train, KS_val1, KS_val2)
        if best<KS_val2:
            best = KS_val2
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


