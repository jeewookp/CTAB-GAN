import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import random
from scipy.stats import ks_2samp
import shutil

def ks_stat(y, yhat):
    return ks_2samp(yhat[y==1], yhat[y!=1]).statistic

class SoftOrdering1DCNN(nn.Module):

    def __init__(self, input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32,
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        hidden_size = sign_size * cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size // 2
        output_size = (sign_size // 4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = nn.Conv1d(
            cha_input,
            cha_input * K,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_input,
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input * K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input * K,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x = x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x

exp_name = 'test'
save_dir = f'./result/{exp_name}'
os.makedirs('./result',exist_ok=True)
os.mkdir(save_dir)

df = pd.read_csv('./preprocessed_df(1).zip')

target_name = 'Target_HBS'
meta_cols = ['val_gb_new', 'sil_2', 'WW', 'hf_score_final', 'num']

dev_df = df[df.val_gb_new==0].reset_index(drop=True)
val_df = df[df.val_gb_new==1].reset_index(drop=True)
test_df = df[df.val_gb_new==2].reset_index(drop=True)

x_train = dev_df.drop(meta_cols, axis=1)
y_train = x_train.pop(target_name)

x_val = val_df.drop(meta_cols, axis=1)
y_val = x_val.pop(target_name)

x_test = test_df.drop(meta_cols, axis=1)
y_test = x_test.pop(target_name)

input_features = x_train.columns

train_tensor_dset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float),
    torch.tensor(y_train.values.reshape(-1,1), dtype=torch.float))

valid_tensor_dset = TensorDataset(torch.tensor(x_val.values, dtype=torch.float),
    torch.tensor(y_val.values.reshape(-1,1), dtype=torch.float))

test_tensor_dset = TensorDataset(torch.tensor(x_test.values, dtype=torch.float),
    torch.tensor(y_test.values.reshape(-1,1), dtype=torch.float))

train_loader = DataLoader(train_tensor_dset, batch_size=2048, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_tensor_dset, batch_size=2048, shuffle=False, num_workers=0)

model = SoftOrdering1DCNN(input_dim=len(input_features), output_dim=1, sign_size=16, cha_input=64,
    cha_hidden=64, K=2, dropout_input=0.3, dropout_hidden=0.3, dropout_output=0.2)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

best = 0
for epoch in range(30):
    for i, (X, y) in enumerate(train_loader):
        model.train()
        y_hat = model.forward(X)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%10==0:
            print(epoch,i,loss.item())
            with open(f'{save_dir}/log.log', "w") as f:
                f.write(f'{epoch} {i} {loss.item()}\n')

    model.eval()
    with torch.no_grad():
        pred_val = model(torch.Tensor(x_val.values))
    pred_val = pred_val.squeeze(1).detach().numpy()
    pred_val = np.exp(pred_val) / (np.exp(pred_val) + 1)

    ks_val = ks_stat(y_val, pred_val)

    print(ks_val)
    with open(f'{save_dir}/log.log', "w") as f:
        f.write(f'{ks_val}\n')

    if ks_val>best:
        best = ks_val
        print('@@@@@@@@@@@@@@@@@@@@@')
        with open(f'{save_dir}/log.log', "w") as f:
            f.write(f'@@@@@@@@@@@@@@@@@@@@@\n')
        best_model = copy.deepcopy(model)

    scheduler.step(ks_val)



with torch.no_grad():
    pred_dev = best_model(torch.Tensor(x_train.values))
    pred_val = best_model(torch.Tensor(x_val.values))
    pred_test = best_model(torch.Tensor(x_test.values))

pred_dev = pred_dev.squeeze(1).detach().numpy()
pred_dev = np.exp(pred_dev)/(np.exp(pred_dev)+1)

pred_val = pred_val.squeeze(1).detach().numpy()
pred_val = np.exp(pred_val)/(np.exp(pred_val)+1)

pred_test = pred_test.squeeze(1).detach().numpy()
pred_test = np.exp(pred_test)/(np.exp(pred_test)+1)



ks_dev = ks_stat(y_train, pred_dev)
ks_val = ks_stat(y_val, pred_val)
ks_test = ks_stat(y_test, pred_test)

print(ks_dev)
print(ks_val)
print(ks_test)



