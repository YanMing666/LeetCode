import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook
import lightgbm as lgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
item   = pd.read_csv('./Antai_AE_round1_item_attr_20190626.csv')
submit = pd.read_csv('./Antai_AE_round1_submit_20190715.csv', header=None)
test   = pd.read_csv('./Antai_AE_round1_test_20190626.csv')
train  = pd.read_csv('./Antai_AE_round1_train_20190626.csv')
def get_processing(df_):
    df = df_.copy()
    df['year'] = df['create_order_time'].apply(lambda x:int(x[0:4]))
    df['month'] = df['create_order_time'].apply(lambda x:int(x[5:6]))
    df['day'] = df['create_order_time'].apply(lambda x: int(x[7:9]))
    df['hour'] = df['create_order_time'].apply(lambda x: int(x[11:12]))
    df['date'] = (df['month'].values - 7) * 31 + df['day']
    del df['create_order_time']
    return df
train = get_processing(train)
print(train)