#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import talos as ta

plt.style.use('ggplot')

### โหลดข้อมูล Timeseries Dataset file 

# num_name = ['46343', '759667', '781756', '844359', '1066528']
# num_name = ['46343', '844359']
# fea_name = [ 'acceleration', 'heartrate', 'steps', 'labeled_sleep']

num_name = ['46343', '781756', '1066528', '844359']
fea_name = [ 'acceleration', 'heartrate', 'labeled_sleep']

df_list = list()
merge_fea_col = pd.DataFrame()

for n in num_name :
  fea_col = list()
  for idx_fea,f in enumerate(fea_name, 0): 
    fea_df = pd.read_csv(root_dir + '/Sleep_MiNiSet/' + n + '_' + f + '.txt', sep=" ", header=None)
    if len(fea_df.columns) < 2 :
      fea_df = pd.read_csv(root_dir + '/Sleep_MiNiSet/' + n + '_' + f + '.txt', sep=",", header=None)
    fea_df['uts_' + f] = pd.to_datetime(fea_df[0], unit='s', origin=pd.Timestamp('2019-10-04'))
    if len(fea_df.columns) <= 3 :
      fea_df.columns = ['time_sec_'+f , f, 'uts_' + f]
    else :
      fea_df.columns = ['time_sec_'+f , f+'_x', f+'_y', f+'_z', 'uts_' + f]  
    fea_col.append(fea_df)

  limit_sec = fea_col[len(fea_name) - 1].iloc[-1:, 0].values[0]
  for num_df in range(len(fea_name)) : 
    fea_col[num_df].set_index('uts_' + fea_name[num_df], inplace=True)
    fea_col[num_df] = fea_col[num_df][(fea_col[num_df]['time_sec_' + fea_name[num_df]] >= 0 ) & ( fea_col[num_df]['time_sec_' + fea_name[num_df]] <= limit_sec + 30)]
    fea_col[num_df] = fea_col[num_df].resample('30s').mean()
    fea_col[num_df].drop(fea_col[num_df].columns[0], axis=1, inplace=True)

  df_list.append(pd.concat([fea_col[0], fea_col[1], fea_col[2]], axis=1))

df_all = pd.concat([df_list[0], df_list[1], df_list[2], df_list[3]], sort=False)
df_all.columns = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'heartrate', 'labeled_sleep']
df_all.describe()

###  Preprocess data 

# reset index
df_all.reset_index(inplace=True)
df_all.columns = ['uts', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'heartrate', 'labeled_sleep']

before_clean_df = df_all

# drop duplicate
df_all.drop_duplicates(inplace=True)

# drop row that label = -1
df_all = df_all[df_all['labeled_sleep'] >= 0]

# fill null value with med
df_all.fillna(df_all.median()['acceleration_x': 'heartrate'], inplace=True)

# แทรกข้อมูลในช่วงเวลาที่ขาดหายไป
df_all.interpolate(method='slinear', inplace=True)

# จัดการลดสัญญาณรบกวนในข้อมูลด้วยการทำ Moving Average 
df_all.rolling(2).mean()

df_all.set_index('uts', inplace=True)
df_all.describe()

# visual เปรียบเทียบข้อมูลก่อน Cleaning

before_clean_df.set_index('uts', inplace=True)
before_clean_df[before_clean_df.columns].plot(subplots=True, layout=(2, 3), figsize=(15,10))

g = sns.PairGrid(before_clean_df, hue='labeled_sleep')
g = g.map_offdiag(plt.scatter)
g = g.add_legend()
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.distplot)

# visual เปรียบเทียบข้อมูลหลัง Cleaning

df_all[df_all.columns].plot(subplots=True, layout=(2, 3), figsize=(15,10))

g = sns.PairGrid(df_all, hue='labeled_sleep')
g = g.map_offdiag(plt.scatter)
g = g.add_legend()
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.distplot)

# ทำการ Normalize ข้อมูลด้วยเทคนิค Max-Min Norm
for column in df_all.loc[:, 'acceleration_x':'heartrate'].columns :
    df_all[column] = df_all.apply(lambda row: (row[column] - df_all[column].min()) / (df_all[column].max() - df_all[column].min()), axis=1)
df_all

# สร้างฟังก์ชันสำหรับสร้าง TimeSeries
def process_create_WindowTimeSeries(df, activity_start, activity_len, time_window, n_feature, step_stride):
    df_series = df
    segments = []
    # วนลูปโดยให้ i มีค่าตั้งแต่ row ที่ 0 ถึง ( จำนวน row - time_window) โดยนับทีละ step_stride
    for i in range(0, len(df_series) - time_window, step_stride):
        
        # เก็บค่าของ row ที่ i ถึง i + time_window
        df_series_feature = df_series.iloc[i: i + time_window]   
        segments.append(np.array(df_series_feature))
        
    # ทำการ reshape ให้มีขนาด  ( #ชุด time_series, #time_step, #features ) 
    reshaped_segments = np.asarray(segments).reshape(-1, time_window, n_feature)

    return reshaped_segments

time_step = 3
time_stride = 1
col_name = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'heartrate', 'labeled_sleep']

time_series = process_create_WindowTimeSeries(df_all[col_name], 0, len(col_name), time_step, len(col_name), time_stride)
time_series_2d = time_series.reshape(time_series.shape[0]*time_series.shape[1],time_series.shape[2])

print(time_series.shape)
print('=================================')
print(time_series_2d.shape)

### Prepare Label Ground Truth (y) 

# Init X
col_name = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'heartrate']
X = process_create_WindowTimeSeries(df_all[col_name], 0, len(col_name), time_step, len(col_name), time_stride)
X_2d = X.reshape(X.shape[0]*X.shape[1], X.shape[2])  # convert to 2D

print(X.shape)
print(X_2d.shape)

# Majority vote prepare Label Ground Truth (y)

def majority_3d(time_series=time_series):
  mj_list = list()

  for t in range(time_series.shape[0]) :
    mj_vote = list()
    for r in range(time_series.shape[1]):
      mj_vote.append( time_series[t][r][time_series.shape[2] - 1] )
    mj_counter = Counter(mj_vote)
    mj_list.append(mj_counter.most_common(1)[0][0])
  return mj_list

def majority_2d(time_series=time_series_2d):
  mj_list = list()
  for r in range(time_series.shape[0]) :
    mj_list.append(time_series[r][-1])
  return mj_list


# Init y 1D
y = np.array(majority_3d())
y = y.reshape(-1, 1)

# Onehot
enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()

print(y.shape)
print('=======================================================')

# Init y 2D
y_2d = np.array(majority_2d())
y_2d = y_2d.reshape(-1, 1)

# Onehot
enc = OneHotEncoder()
enc.fit(y_2d)
y_2d = enc.transform(y_2d).toarray()

print(y_2d.shape)

### Prepare training, validation, and test data

# # train test split

# X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_train, X_valid, y_train, y_valid = train_test_split(X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)

# X_2d_t, X_2d_test, y_2d_t, y_2d_test = train_test_split(X_2d, y_2d, test_size=0.2, random_state=42, stratify=y_2d)
# X_2d_train, X_2d_valid, y_2d_train, y_2d_valid = train_test_split(X_2d_t, y_2d_t, test_size=0.2, random_state=42, stratify=y_2d_t)

# X_train = np.expand_dims(X_train, axis=-1)
# X_test = np.expand_dims(X_test, axis=-1)
# X_valid = np.expand_dims(X_valid, axis=-1)

# X_2d_train = np.expand_dims(X_2d_train, axis=-1)
# X_2d_test = np.expand_dims(X_2d_test, axis=-1)
# X_2d_valid = np.expand_dims(X_2d_valid, axis=-1)

# print(X_train.shape)
# print(X_test.shape)
# print(X_valid.shape)

# print('=============================')
# print(X_2d_train.shape)
# print(X_2d_test.shape)
# print(X_2d_valid.shape)

# train test split

X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

print('=============================')


X_2d_train  = X_train.reshape(-1, 3, 4) 
X_2d_valid  = X_valid.reshape(-1, 3, 4) 
X_2d_test  = X_test.reshape(-1, 3, 4) 

print(X_2d_train.shape)
print(X_2d_test.shape)
print(X_2d_valid.shape)

