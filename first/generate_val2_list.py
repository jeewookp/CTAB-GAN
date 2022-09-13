import pandas as pd
import numpy as np
import random

df_test = pd.read_csv('/Users/john/project/CTAB-GAN/first_data/ML_data_val.csv')

val2_selected_indices = random.sample(range(len(df_test)),3000)
np.save('/Users/john/project/CTAB-GAN/first_data/val2_list.npy',sorted(val2_selected_indices))

