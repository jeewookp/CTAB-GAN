import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Conv_Relu_Conv(torch.nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(Conv_Relu_Conv,self).__init__()
        self.fc0 = nn.Linear(in_dims,hidden_dims)
        self.fc1 = nn.Linear(hidden_dims,out_dims)

    def forward(self, input):
        input = F.relu(self.fc0(input))
        input = self.fc1(input)
        return input

class Conv_Relu_Conv2(torch.nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(Conv_Relu_Conv2,self).__init__()
        self.fc0 = nn.Linear(in_dims,hidden_dims)
        self.fc1 = nn.Linear(hidden_dims,hidden_dims)
        self.fc2 = nn.Linear(hidden_dims,out_dims)


    def forward(self, input):
        input = F.relu(self.fc0(input))
        input = F.relu(self.fc1(input))
        input = self.fc2(input)
        return input

hidden_dims = 512
feature_dims = 10
batch_size = 512

df_train = pd.read_csv('/Users/john/project/CTAB-GAN/first_data/ML_data_dev.csv')
df_test = pd.read_csv('/Users/john/project/CTAB-GAN/first_data/ML_data_val.csv')

x_train = torch.from_numpy(df_train.drop(['dev_val', 'pk','bad'], axis=1).values).to(torch.float32)
x_test = torch.from_numpy(df_test.drop(['dev_val', 'pk','bad'], axis=1).values).to(torch.float32)

Y_train = torch.from_numpy(pd.get_dummies(df_train.bad).values[:,0])
Y_test = torch.from_numpy(pd.get_dummies(df_test.bad).values[:,0])

classifier = Conv_Relu_Conv(x_train.shape[1],hidden_dims,2)
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
        output = classifier(x_test)
        output = torch.exp(output)
        output = output[:,0]/(output[:,0]+output[:,1])
        sorted_indices = torch.argsort(output)
        sorted_labels = Y_test[sorted_indices]
        n_positives = torch.cumsum(sorted_labels,dim=0)
        n_negatives = torch.arange(1,n_positives.shape[0]+1) - n_positives
        cum_pos_ratio = n_positives/n_positives[-1]
        cum_neg_ratio = n_negatives/n_negatives[-1]
        KS = torch.max(cum_pos_ratio - cum_neg_ratio)
        print(KS.item())
        if best<KS.item():
            best = KS.item()
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


x_train_gan = torch.cat([x_train, Y_train.unsqueeze(1)], dim=1)
x_test_gan = torch.cat([x_test, Y_test.unsqueeze(1)], dim=1)

discriminator = Conv_Relu_Conv2(x_train_gan.shape[1],512,1)
generator = Conv_Relu_Conv2(feature_dims,512,x_train_gan.shape[1])
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)

neg_target = torch.zeros(batch_size,dtype=torch.uint8)
pos_target = torch.ones(batch_size,dtype=torch.uint8)
disc_target = torch.cat([neg_target,pos_target],dim=0)
for i in range(100000):
    discriminator.train()
    generator.train()

    optimizer_d.zero_grad()
    optimizer_g.zero_grad()
    output = discriminator(generator(torch.randn((batch_size, feature_dims), dtype=torch.float32))).squeeze(1)
    loss_g = torch.mean((output-pos_target)**2)
    loss_g.backward()
    optimizer_g.step()

    optimizer_d.zero_grad()
    optimizer_g.zero_grad()
    samples = random.sample(range(x_train_gan.shape[0]),batch_size)
    x_train_gan_unit = x_train_gan[samples]
    with torch.no_grad():
        fake_data = generator(torch.randn((batch_size,feature_dims),dtype=torch.float32))
    batch = torch.cat([fake_data,x_train_gan_unit],dim=0)
    output = discriminator(batch).squeeze(1)
    loss_d = torch.mean((output-disc_target)**2)
    loss_d.backward()
    optimizer_d.step()

    if i==0:
        avg_loss_d = loss_d.item()
        avg_loss_g = loss_g.item()
    else:
        avg_loss_d = 0.999 * avg_loss_d + 0.001 * loss_d.item()
        avg_loss_g = 0.999 * avg_loss_g + 0.001 * loss_g.item()
    if i % 1000 == 0:
        print(i, avg_loss_d, avg_loss_g)




with torch.no_grad():
    fake_data = generator(torch.randn((100,feature_dims),dtype=torch.float32))











