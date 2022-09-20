import pandas as pd
import numpy as np
import random

df = pd.read_csv('/Users/john/project/CTAB-GAN/first_data/hf_model_v2_seg200_backscofing_john.csv')

train = df[df['val_gb_new'] == 0]
val = df[df['val_gb_new'] == 1]
test = df[df['val_gb_new'] == 2]

x0_val_indices = random.sample(list(train.index),int(len(train)*0.2))
np.save('/Users/john/project/CTAB-GAN/first_data/x0_val.npy',sorted(x0_val_indices))

x1_val_indices = random.sample(list(val.index),int(len(val)*0.2))
np.save('/Users/john/project/CTAB-GAN/first_data/x1_val.npy',sorted(x1_val_indices))