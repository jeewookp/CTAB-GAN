import pandas as pd
import sweetviz as sv
import numpy as np

df_fake = pd.read_csv('/Users/john/project/CTAB-GAN/first_data/ML_data_fake.csv').values
for i in range(100):
    print(np.min(np.sum(np.abs(df_fake[i:i+1]-df_fake[list(set(range(df_fake.shape[0]))-set([i]))]),axis=1)))



df_fake = pd.read_csv("/Users/john/project/CTAB-GAN/first_data/ML_data_fake.csv")
advert_report_fake = sv.analyze(df_fake)
advert_report_fake.show_html('./sweetviz_Advertising_fake.html')

df_train = pd.read_csv("/Users/john/project/CTAB-GAN/first_data/ML_data_dev.csv")
advert_report_train = sv.analyze(df_train)
advert_report_train.show_html('./sweetviz_Advertising_train.html')

df1 = sv.compare(df_train, df_fake)
df1.show_html('./sweetviz_Compare.html')

total_number = len(df_train)
for key in df_fake.keys():
    psi = 0
    for value in df_fake[key].value_counts().keys():
        train_ratio = df_train[key].value_counts()[value]/total_number
        fake_ratio = df_fake[key].value_counts()[value]/total_number
        psi += (train_ratio-fake_ratio)*np.log(train_ratio/fake_ratio)
    print(key,psi)


