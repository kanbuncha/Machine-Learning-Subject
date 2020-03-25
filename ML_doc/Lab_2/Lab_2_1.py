# ตอนที่1: การทดลองเตรียมข้อมูล ปรับค่าข้อมูล และจัดแบ่งชุด Train, Test เพื่อสอนโมเดล
# 1.1 เตรียมข้อมูลทดลอง

# Stock data
import quandl
import datetime

# Analyzing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics

stock_MSFT = pd.read_csv('MSFT.csv')
stock_MSFT.set_index('Date', inplace=True)

# stock_MSFT = quandl.get("WIKI/MSFT")
# stock_MSFT.to_csv('MSFT.csv')
# stock_MSFT.index = pd.to_datetime(stock_MSFT.index)
stock_MSFT

## 1.2 ปรับรูปแบบของข้อมูล 

# สร้างข้อมูลทางเลือกด้วยการทำ Normalization ข้อมูล ‘Close’ ที่เลือก
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
stock_MSFT['Close_norm'] = scaler.fit_transform(np.array(stock_MSFT['Close']).reshape(-1, 1))
stock_MSFT_Close = pd.DataFrame(stock_MSFT['Close_norm'])

# สร้างข้อมูลราคาวันถัดไป Next_N-day เพื่อใช้เป็นคำตอบ (Ground Truth) ในการคาดการณ์
Next_N_day = 30
GT = stock_MSFT_Close.iloc[Next_N_day :]
stock_MSFT = stock_MSFT_Close.iloc[:-Next_N_day]
stock_MSFT = stock_MSFT.assign(GT=GT.values)
stock_MSFT

## 1.3 จัดเตรียมข้อมูลสำหรับ train validation และ test


# ข้อมูล test ให้ใช้ข้อมูล ‘GT’ ช่วง 60 วันท้าย
test_df = stock_MSFT.iloc[len(stock_MSFT) - 60:]
X_test = np.array(test_df.Close_norm).reshape(-1, 1)
y_test = test_df.GT

# # เตรียมข้อมูล train, validate
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
    np.array(stock_MSFT.Close_norm[:-60]).reshape(-1, 1),
    stock_MSFT.GT[:-60],
    test_size=0.2, random_state=42,
    shuffle=True)

## 1.4 แสดงรูปกราฟการกระจายของ train validate ที่แบ่งจากข้อ 1.3

# ตั้งค่าขนาดพื้นที่ภาพ
plt.figure(figsize=(30,10))

# scatter plot ความสัมพันธ์ของค่า X_train, y_train 
plt.scatter(X_train, y_train, marker='o', color='blue', label='Training set')
plt.scatter(X_valid, y_valid, marker='o', color='red', label='Validate set')
plt.scatter(X_test, y_test, marker='o', color='green', label='Testing set')
plt.title('Train Validate and Test dataset')
plt.legend()
plt.show()