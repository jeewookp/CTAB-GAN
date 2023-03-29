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


class DNN(nn.Module):

    def __init__(self, input_dim, output_dim, nn_depth, nn_width, dropout, momentum):
        super().__init__()

        self.bn_in = nn.BatchNorm1d(input_dim, momentum=momentum)
        self.dp_in = nn.Dropout(dropout)
        self.ln_in = nn.Linear(input_dim, nn_width, bias=False)

        self.bnorms = nn.ModuleList([nn.BatchNorm1d(nn_width, momentum=momentum) for i in range(nn_depth - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(nn_depth - 1)])
        self.linears = nn.ModuleList([nn.Linear(nn_width, nn_width, bias=False) for i in range(nn_depth - 1)])

        self.bn_out = nn.BatchNorm1d(nn_width, momentum=momentum)
        self.dp_out = nn.Dropout(dropout / 2)
        self.ln_out = nn.Linear(nn_width, output_dim, bias=False)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.bn_in(x)
        x = self.dp_in(x)
        x = nn.functional.relu(self.ln_in(x))

        for bn_layer, dp_layer, ln_layer in zip(self.bnorms, self.dropouts, self.linears):
            x = bn_layer(x)
            x = dp_layer(x)
            x = ln_layer(x)
            x = nn.functional.relu(x)

        x = self.bn_out(x)
        x = self.dp_out(x)
        x = self.ln_out(x)
        return x

if __name__=='__main__':
    exp_name = '0321_dnn'
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

    model = DNN(
        input_dim=len(input_features),
        output_dim=1,
        nn_depth=3,
        nn_width=256,
        dropout=0.2,
        momentum=0.1
    ).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best = 0
    for epoch in range(30):
        for i, (X, y) in enumerate(train_loader):
            model.train()
            y_hat = model.forward(X.cuda())
            loss = criterion(y_hat, y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10==0:
                print(epoch,i,loss.item())
                with open(f'{save_dir}/log.log', "a") as f:
                    f.write(f'{epoch} {i} {loss.item()}\n')

        model.eval()
        with torch.no_grad():
            pred_val = model(torch.Tensor(x_val.values).cuda())
        pred_val = pred_val.squeeze(1).cpu().detach().numpy()
        pred_val = np.exp(pred_val) / (np.exp(pred_val) + 1)

        ks_val = ks_stat(y_val, pred_val)

        print(ks_val)
        with open(f'{save_dir}/log.log', "a") as f:
            f.write(f'{ks_val}\n')

        if ks_val>best:
            best = ks_val
            print('@@@@@@@@@@@@@@@@@@@@@')
            with open(f'{save_dir}/log.log', "a") as f:
                f.write(f'@@@@@@@@@@@@@@@@@@@@@\n')
            best_model = copy.deepcopy(model)

        scheduler.step(ks_val)



    with torch.no_grad():
        pred_dev = best_model(torch.Tensor(x_train.values).cuda())
        pred_val = best_model(torch.Tensor(x_val.values).cuda())
        pred_test = best_model(torch.Tensor(x_test.values).cuda())

    pred_dev = pred_dev.squeeze(1).cpu().detach().numpy()
    pred_dev = np.exp(pred_dev)/(np.exp(pred_dev)+1)

    pred_val = pred_val.squeeze(1).cpu().detach().numpy()
    pred_val = np.exp(pred_val)/(np.exp(pred_val)+1)

    pred_test = pred_test.squeeze(1).cpu().detach().numpy()
    pred_test = np.exp(pred_test)/(np.exp(pred_test)+1)



    ks_dev = ks_stat(y_train, pred_dev)
    ks_val = ks_stat(y_val, pred_val)
    ks_test = ks_stat(y_test, pred_test)

    print(ks_dev)
    print(ks_val)
    print(ks_test)

    with open(f'{save_dir}/log.log', "a") as f:
        f.write(f'{ks_dev} {ks_val} {ks_test}\n')

