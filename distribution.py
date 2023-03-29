import torch
from resnet import SoftOrdering1DCNN
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

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


model = SoftOrdering1DCNN(input_dim=len(input_features), output_dim=1, sign_size=16, cha_input=3,
    dropout_input=0.3, dropout_output=0.2).cuda()
model.load_state_dict(torch.load('./result/0327_resnet/model.pt'))
model.eval()

with torch.no_grad():
    input = torch.Tensor(x_train.values).cuda()
    pred_dev = []
    for i in range((input.shape[0] - 1) // 1000 + 1):
        print(i)
        pred_dev.append(model(input[i * 1000:(i + 1) * 1000]))
    pred_dev = torch.cat(pred_dev, dim=0)

    input = torch.Tensor(x_val.values).cuda()
    pred_val = []
    for i in range((input.shape[0] - 1) // 1000 + 1):
        print(i)
        pred_val.append(model(input[i * 1000:(i + 1) * 1000]))
    pred_val = torch.cat(pred_val, dim=0)

    input = torch.Tensor(x_test.values).cuda()
    pred_test = []
    for i in range((input.shape[0] - 1) // 1000 + 1):
        print(i)
        pred_test.append(model(input[i * 1000:(i + 1) * 1000]))
    pred_test = torch.cat(pred_test, dim=0)

pred_dev = pred_dev.squeeze(1).cpu().numpy()
pred_val = pred_val.squeeze(1).cpu().numpy()
pred_test = pred_test.squeeze(1).cpu().numpy()

pred_dev = np.exp(pred_dev) / (np.exp(pred_dev) + 1)
pred_val = np.exp(pred_val) / (np.exp(pred_val) + 1)
pred_test = np.exp(pred_test) / (np.exp(pred_test) + 1)

np.save('./result/0327_resnet/result.npy',{'pred_dev':pred_dev,'pred_val':pred_val,'pred_test':pred_test})

