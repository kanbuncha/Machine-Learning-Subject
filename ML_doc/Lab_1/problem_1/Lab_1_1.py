#!/usr/bin/env python
# coding: utf-8

# # Ploblem: 1

# In[1]:


# problem 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'  #ความละเอียด 2เท่า")


# อ่านข้อมูล Dataset (.csv) ให้อ่านข้อมูลเข้ามาอยู่ในรูปของ dataframe 
df = pd.read_csv('watch_test2_sample.csv')


# แสดงรายละเอียดพื้นฐาน Dataset
df.info()

# แสดงรายละเอียดทางสถิติของ Dataset
df.describe()

## Data cleaning

### null value and duplicate

# 1 null value and duplicate
# ลบแถวที่มีข้อมูลซ้ำกัน 
df.drop_duplicates(inplace=True)

# reset index
df.reset_index(drop=True, inplace=True)

# ตรวจสอบว่าใน column gyro.x,gyro.y และ gyro.z มีค่าเป็น null value หรือไม่
df[(df['gyro.x'].isnull()) | (df['gyro.y'].isnull()) | (df['gyro.z'].isnull())]

# ลบ null value ใน column gyro.x, gyro.y และ gyro.z
df.dropna(subset=['gyro.x', 'gyro.y', 'gyro.z'], inplace=True)

# ตรวจสอบว่าใน column accelerateX, accelerateY, accelerateZ และ compass มีค่าเป็น null value หรือไม่
df[(df['accelerateX'].isnull()) | (df['accelerateY'].isnull()) | (df['accelerateZ'].isnull()) |(df['compass'].isnull())]

# ลบ null value ใน column accelerateX, accelerateY, accelerateZ และ compass
df.dropna(subset=['accelerateX', 'accelerateY', 'accelerateZ', 'compass'], thresh=2, inplace=True)

# แทน null value ด้วยค่าเฉลี่ยของแต่ละ column 
df.fillna(df.mean()['accelerateX':'accelerateZ'], inplace=True)

# แทน null value ด้วยค่าเฉลี่ยของแต่ละ column 
df.fillna(df.mean()['heartrate':'pressure'], inplace=True)

# แทนค่า 0 ด้วยค่าเฉลี่ยของแต่ละ column (heartrate, pressure)
df.replace({
    'heartrate': {0: df['heartrate'].mean()},
    'pressure': {0: df['pressure'].mean()},
}, inplace=True)

# แทนค่า 0 ด้วยข้อมูลของ row ก่อนหน้า ใน column gps.x และ gps.y
df[["gps.x", "gps.y"]]= df[["gps.x", "gps.y"]].replace(0, method='ffill')

# แทนค่าที่มีค่าโดดเกินค่าที่กำหนดด้วยค่าเฉลี่ยของแต่ละ column (heartrate, light, compass)
df['heartrate'] = df['heartrate'].apply(lambda row: df['heartrate'].mean() if row > 150 else row)
df['light'] = df['light'].apply(lambda row: df['light'].mean() if row > 1000 else row)
df['compass'] = df['compass'].apply(lambda row: df['compass'].mean() if row > 400 else row)

df.reset_index(drop=True, inplace=True)

df.describe()

before_resample = df

# แยกช่วงเวลาที่ช่วงเวลาห่างกันเยอะ เป็น 3 ช่วง
time_1 = df[:33]
time_2 = df[33:239]
time_3 = df[239:]
print(df.shape)
print(time_1.shape[0], time_2.shape[0], time_3.shape[0])

# resample ทั้ง 3 ช่วงเวลา
def resample_df(df):
    df['uts'] = pd.to_datetime(df['uts'])
    df.set_index('uts', inplace=True)
    df = df.resample('20s').sum()
    print(df.shape)
    return df

resample_1 = resample_df(time_1.reset_index(drop=True))
resample_2 = resample_df(time_2.reset_index(drop=True))
resample_3 = resample_df(time_3.reset_index(drop=True))

# ทำการ concat ทั้ง 3 ช่วงเวลา
df = pd.concat([resample_1, resample_2, resample_3])

###  interpolate

# แทรกข้อมูลในช่วงเวลาที่ขาดหายไป
df.interpolate(method='slinear', inplace=True)


###  rolling

# จัดการลดสัญญาณรบกวนในข้อมูลด้วยการทำ Moving Average 
df.rolling(2).mean()

problem_2 = df

## Data Normalization (Max-Min normolization)

# ทำการ Normalize ข้อมูลด้วยเทคนิค Max-Min Norm
for column in df.loc[:, 'accelerateX':].columns :
    df[column] = df.apply(lambda row: (row[column] - df[column].min()) / (df[column].max() - df[column].min()), axis=1)
df

df.describe()

## 1.2 Visualiztion

### Line plot

# แสดงกราฟข้อมูลแต่ละ feature (Column) ด้วย Line Plot 
df[df.columns].plot(subplots=True, layout=(5, 3), figsize=(15,18))

### 3D plot

# แสดงกราฟข้อมูลความสัมพันธ์ระหว่างคู่ features ด้วย 3D Scatter Plot
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(df.accelerateX, df.accelerateY, df.accelerateZ, c='cyan', s=20, edgecolor='k')
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_xlabel('z')

ax.view_init(30, -60)


ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(df['gyro.x'], df['gyro.y'], df['gyro.z'], c='violet', s=20, edgecolor='k')
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_xlabel('z')

ax.view_init(30, -60)

plt.show()

### Geolocation

before_resample.describe()

# Export ภาพแผนที่จาก "https://www.openstreetmap.org/export#map=6/12.780/102.986"

# อ่านไฟล์ภาพแผนที่
map_im = plt.imread('map.png')

fig, ax = plt.subplots(figsize=(15, 10))

# plot ตำแหน่งพิกัด GPS
ax.scatter(before_resample['gps.x'], before_resample['gps.y'], zorder=1, alpha=0.5, c='r', s=20)
ax.set_title('Plotting Spatial Data on Map')

# lat_min, lat_max, long_min, long_max
BBox =[13.5392, 13.6301, 100.2535, 100.3768]

print(BBox)

ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])
ax.imshow(map_im, zorder=0, extent=BBox, aspect='auto')

## 1.3 Shape

# แปลง Dataframe เป็น array
array_1 = np.array(df[['accelerateX', 'accelerateY', 'accelerateZ', 'compass', 'heartrate']])
array_1.shape

array_1

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
col_name = ['accelerateX', 'accelerateY', 'accelerateZ', 'compass', 'heartrate']

time_series = process_create_WindowTimeSeries(df[col_name], 0, len(col_name), time_step, len(col_name), time_stride)
print(time_series.shape)
print(time_series)

# ปรับอะเรย์ 3 มิติที่ได้ ให้อยู่ในรูปของ 2 มิติขนาด ( #ชุด*#time_step, #features )
time_series_2d = time_series.reshape(time_series.shape[0]*time_series.shape[1],time_series.shape[2])
print(time_series_2d.shape)
print(time_series_2d)

# แสดงภาพของอะเรย์ชุดที่ 1
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Data')
ax.imshow(array_1, aspect='auto')

# แสดงภาพของอะเรย์ชุดที่ 2 (TimeSeries)
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Data')
ax.imshow(time_series_2d, aspect='auto')


# In[2]:


# problem 2
df = problem_2[['accelerateX', 'accelerateY', 'accelerateZ']]

df.describe()

# ปรับให้เป็น zero mean 
for column in df.columns :
    df[column] = df.apply(lambda row: (row[column] - df[column].mean()), axis=1)
df

# คำนวณค่า covariance matrix ของชุดข้อมูล df
covariance_matrix = np.dot(df.T, df) / (len(df) -1)
covariance_matrix

### 2.2 eigenvalue and eigenvecter

# คำนวณค่า eigenvalue / eigenvector จาก covariance matrix 
eigenvalue, eigenvecter = np.linalg.eig(covariance_matrix)
print('eigenvalue:', eigenvalue)
print('eigenvecter:', eigenvecter)

###  2.3 Graph

# แสดงกราฟแท่ง (Bar graph) ของค่า Eigenvalue ที่จัดเรียงค่าจากมากไปน้อย
plt.bar(np.arange(3), height=np.sort(eigenvalue)[::-1], width=0.8, align='center', edgecolor='k')

# แสดงปรับขนาดของ Eigenvector ด้วยค่า Eigenvalue
ev1 = eigenvecter[: ,0] * np.sqrt(eigenvalue[0])
ev2 = eigenvecter[: ,1] * np.sqrt(eigenvalue[1])
ev3 = eigenvecter[: ,2] * np.sqrt(eigenvalue[2])
print(ev1, ev2, ev3)

# แสดงกราฟความสัมพันธ์ของ feature และ eigen vector
fig = plt.figure(figsize=(35, 8))

ax = fig.add_subplot(141, projection='3d')
ax.plot(df['accelerateX'], df['accelerateY'], df['accelerateZ'], 'o', markersize=10, color='green', alpha=0.2)
ax.plot([df['accelerateX'].mean()], [df['accelerateY'].mean()], [df['accelerateZ'].mean()], 'o', markersize=10, color='red', alpha=0.5)

ax.plot([0, ev1[0]], [0, ev1[1]], [0, ev1[2]], color='red', alpha=0.8, lw=2)
ax.plot([0, ev2[0]], [0, ev2[1]], [0, ev2[2]], color='violet', alpha=0.8, lw=2)
ax.plot([0, ev3[0]], [0, ev3[1]], [0, ev3[2]], color='cyan', alpha=0.8, lw=2)

ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_xlabel('z')

ax.view_init(30, -60)

plt.show()

### 2.4 PCA

print('eigenvalue:', eigenvalue)
print('eigenvecter:', eigenvecter)

# ลดมิติของข้อมูลจาก 3D features 𝑥 (accelerateX, accelerateY, accelerateZ) ลงเหลือ 2D โดยเลือก eigenvector 2 vector แรกที่มีค่าสูงสุด
x_pca = np.dot(np.take(eigenvecter,np.argsort(eigenvalue)[1:].tolist(),axis=0), df.T)
df_x_pca = pd.DataFrame(x_pca)
df_x_pca

# แสดงภาพ df_x_pca ด้วย heatmap
sns.heatmap(df_x_pca)


# ### 2.1 covariance_matrix

# # Problem 2
