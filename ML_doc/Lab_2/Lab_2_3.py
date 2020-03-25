# ตอนที่ 3: การทดลองการค้นหาพารามิเตอร์ที่ดีที่สุดสำหรับโมเดล
# 3.1 กำหนดรายการพารามิเตอร์ทั้งหมดที่ต้องการทดสอบหาค่าที่ดีที่สุดของโมเดล SVC

# แสดงรูปกราฟเปรียบเทียบ score ที่ได้จากโมเดลทั้ง 4 แบบที่คำนวณข้างบน

from matplotlib.ticker import MaxNLocator
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(np.arange(len(LRM_cross)), LRM_cross, color='red', alpha=0.8, lw=2)
ax.plot(np.arange(len(LRM_cross)), svr_lin_cross , color='green', alpha=0.8, lw=2)
ax.plot(np.arange(len(LRM_cross)), svr_rbf_cross , color='blue', alpha=0.8, lw=2)
ax.plot(np.arange(len(LRM_cross)), svr_poly_cross , color='yellow', alpha=0.8, lw=2)

ax.legend(['LRM', 'svr_lin', 'svr_rbf', 'svr_poly'])
ax.set_xlabel('K-Fold')
ax.set_ylabel('Accuracy')
plt.title('Accuracy of each K-Fold model ')
plt.show()

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly')
y_pos = np.arange(len(objects))
performance = [LRM_cross.mean(), svr_lin_cross.mean(), svr_rbf_cross.mean(), svr_poly_cross.mean()]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Mean')
plt.xlabel('Model')
plt.title('Score Mean')

plt.show()

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly')
y_pos = np.arange(len(objects))
performance = [LRM_cross.std(), svr_lin_cross.std(), svr_rbf_cross.std(), svr_poly_cross.std()]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('std')
plt.xlabel('Model')
plt.title('Score STD.')

plt.show()

## 2.2 ทดสอบโมเดลทั้ง 4 แบบ ที่กำหนดพารามิเตอร์ไว้ในข้อ 2.1

# ทำการ train โมเดลทั้ง 4 แบบ ด้วยข้อมูล Train ที่แบ่งไว้
LRM.fit(X_train, y_train)
svr_lin.fit(X_train, y_train)
svr_rbf.fit(X_train, y_train)
svr_poly.fit(X_train, y_train)

# ทำการ predict ข้อมูลชุด Validation และ Test
LRM_pred_valid = LRM.predict(X_valid)
LRM_pred_test  = LRM.predict(X_test)

svr_lin_pred_valid = svr_lin.predict(X_valid)
svr_lin_pred_test = svr_lin.predict(X_test)

svr_rbf_pred_valid = svr_rbf.predict(X_valid)
svr_rbf_pred_test = svr_rbf.predict(X_test)

svr_poly_pred_valid = svr_poly.predict(X_valid)
svr_poly_pred_test = svr_poly.predict(X_test)

# คำนวณค่าตัววัดประสิทธิภาพของการทำนายจากโมเดลทั้ง 4 แบบ โดยวัดค่า MSE และ R2
from sklearn import metrics

def perfomance_measure(model_name, model_pred_valid, model_pred_test) :
    msr_valid = metrics.mean_squared_error(y_valid, model_pred_valid)
    msr_test = metrics.mean_squared_error(y_test, model_pred_test)
    r2_valid = metrics.r2_score(y_valid, model_pred_valid)
    r2_test = metrics.r2_score(y_test, model_pred_test)
    
    print('=== ', model_name, ' ===')
    print('Mean Squared Error LRM validation set:', msr_valid)
    print('Mean Squared Error LRM test set:', msr_test)
    print('R2 LRM validation set: ', r2_valid)
    print('R2 LRM test set: ', r2_test)
    print('\n')
    
    return msr_valid, msr_test, r2_valid, r2_test

LRM_msr_valid, LRM_msr_test, LRM_r2_valid, LRM_r2_test = perfomance_measure('LRM', LRM_pred_valid, LRM_pred_test)
svr_lin_msr_valid, svr_lin_msr_test, svr_lin_r2_valid, svr_lin_r2_test = perfomance_measure('svr_lin', svr_lin_pred_valid, svr_lin_pred_test)
svr_rbf_msr_valid, svr_rbf_msr_test, svr_rbf_r2_valid, svr_rbf_r2_test = perfomance_measure('svr_rbf', svr_rbf_pred_valid, svr_rbf_pred_test)
svr_poly_msr_valid, svr_poly_msr_test, svr_poly_r2_valid, svr_poly_r2_test = perfomance_measure('svr_poly', svr_poly_pred_valid, svr_poly_pred_test)

# Train set graph
plt.figure(1, figsize=(30,10))
plt.title('Linear Regression | Price vs Time')
plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.scatter(X_train, LRM.predict(X_train), color='r', label='LRM')
plt.scatter(X_train, svr_lin.predict(X_train), color='k', label='svr_lin')
plt.scatter(X_train, svr_rbf.predict(X_train), color='g', label='svr_rbf')
plt.scatter(X_train, svr_poly.predict(X_train), color='m', label='svr_poly')
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# แสดงรูปกราฟเปรียบเทียบผลการ Predict validation และ Predict test ข้างต้นจากโมเดลทั้ง 4 

objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly')
y_pos = np.arange(len(objects))
performance = [LRM_msr_valid, svr_lin_msr_valid, svr_rbf_msr_valid, svr_poly_msr_valid]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Mean Squared Error')
plt.xlabel('Model')
plt.title('Mean Squared Error validation set')

plt.show()

objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly')
y_pos = np.arange(len(objects))
performance = [LRM_msr_test, svr_lin_msr_test, svr_rbf_msr_test, svr_poly_msr_test]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Mean Squared Error')
plt.xlabel('Model')
plt.title('Mean Squared Error test set')

plt.show()

objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly')
y_pos = np.arange(len(objects))
performance = [LRM_r2_valid, svr_lin_r2_valid, svr_rbf_r2_valid, svr_poly_r2_valid]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('R2')
plt.xlabel('Model')
plt.title('R2 validation set')

plt.show()

objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly')
y_pos = np.arange(len(objects))
performance = [LRM_r2_test, svr_lin_r2_test, svr_rbf_r2_test, svr_poly_r2_test]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('R2')
plt.xlabel('Model')
plt.title('R2 test set')

plt.show()

k_Fold = 10 # เลือก 10
c_param = [1, 10, 100, 1000]          # เลือกค่า 2, 10, 100, 500
gamma = [0.01, 0.5, 1.0]               # เลือกค่า 0.3, 0.5, 1.0
svc_kernel = ('linear', 'rbf', 'poly')

tuned_parameters = {'kernel': svc_kernel, 'C': c_param, 'gamma': gamma}

## 3.2 เตรียมการค้นหาพารามิเตอร์ที่ดีที่สุดโดยใช้ฟังก์ชั่น GridSearchCV 

# กำหนดโมเดล
model = SVR()

# ใช้ cross validation (cv) เป็น kfold ที่กำหนดไว้ในตอนข้อ 2.1
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(model, tuned_parameters, cv=kf, scoring='r2')

# นำค่าพารามิเตอร์ที่ดีที่สุดที่ได้จาก GridSearchCV() ไปสอนโมเดล แสดงค่า score และพารามิเตอร์ที่ดีที่สุด fit() / best_params_/ best_score
clf.fit(X_train, y_train)
sorted(clf.cv_results_.keys())
best_score = clf.best_score_
best_params = clf.best_params_
print(best_params)

# save ผลลัพธ์จากการทำ GridSearchCV cv_results ลงบนไฟล์ .csv
best_param_df = pd.DataFrame(list(best_params.items()), columns=['Parameter', 'Cost'])
best_param_df.to_csv('cv_results.csv', index=False)

## 3.3 ทำการ predict ข้อมูลชุด Validation และ Test

cv_pred_valid = clf.predict(X_valid)
cv_pred_test  = clf.predict(X_test)

print('cv_pred_valid: ', cv_pred_valid)
print('cv_pred_test: ', cv_pred_test)

## 3.4 คำนวณค่าตัววัดประสิทธิภาพของการทำนายที่ได้จากข้อ 3.5 โดยวัดค่า MSE และ R2

cv_msr_valid, cv_msr_test, cv_r2_valid, cv_r2_test = perfomance_measure('cv_pred_test', cv_pred_valid, cv_pred_test)

## 3.5 แสดงรูปกราฟเปรียบเทียบผลการ Predict validation และ Predict test ข้างต้นจากโมเดลทั้ง 4 แบบ โดยในรูปแบบกราฟที่แสดงเห็นความแตกต่างชัดเจน เช่น กราฟ plot, bar, scatter เป็นต้น

# Train set graph
plt.figure(1, figsize=(30,15))
plt.title('Predict Test set')
plt.scatter(X_test, y_test, edgecolor='wheat', label='Test Data')
plt.plot(X_test, cv_pred_test, color='cyan',label='cv')
plt.plot(X_test, LRM_pred_test, color='lime', label='LRM')
plt.plot(X_test, svr_lin_pred_test, color='gold', label='SVR_LIN')
plt.plot(X_test, svr_rbf_pred_test, color='coral', label='SVR_RBF')
plt.plot(X_test, svr_poly_pred_test, color='teal', label='SVR_POLY')

plt.xlabel('y predict')
plt.ylabel('y true')
plt.legend()
plt.show()

# Train set graph
plt.figure(1, figsize=(30,15))
plt.title('Predict Test set')
plt.scatter(X_valid, y_valid, color='teal', label='Test data')
plt.scatter(X_valid, cv_pred_valid, color='cyan',label='cv')
plt.scatter(X_valid, LRM_pred_valid, color='lime', label='LRM')
plt.scatter(X_valid, svr_lin_pred_valid, color='gold', label='SVR_LIN')
plt.scatter(X_valid, svr_rbf_pred_valid, color='coral', label='SVR_RBF')
plt.scatter(X_valid, svr_poly_pred_valid, color='coral', label='SVR_POLY')

plt.xlabel('y predict')
plt.ylabel('y true')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))


# Use the pyplot interface to change just one subplot...
plt.sca(axes[0, 0])
objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly', 'cv')
y_pos = np.arange(len(objects))
performance = [LRM_msr_valid, svr_lin_msr_valid, svr_rbf_msr_valid, svr_poly_msr_valid, cv_msr_valid]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=45)
plt.ylabel('Mean Squared Error')
plt.xlabel('Model')
plt.title('Mean Squared Error validation set')

plt.sca(axes[0, 1])
objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly', 'cv')
y_pos = np.arange(len(objects))
performance = [LRM_msr_test, svr_lin_msr_test, svr_rbf_msr_test, svr_poly_msr_test, cv_msr_test]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=45)
plt.ylabel('Mean Squared Error')
plt.xlabel('Model')
plt.title('Mean Squared Error test set')

plt.sca(axes[1, 0])
objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly', 'cv')
y_pos = np.arange(len(objects))
performance = [LRM_r2_valid, svr_lin_r2_valid, svr_rbf_r2_valid, svr_poly_r2_valid, cv_r2_valid]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=45)
plt.ylabel('R2')
plt.xlabel('Model')
plt.title('R2 validation set')

plt.sca(axes[1, 1])
objects = ('LRM', 'svr_lin', 'svr_rbf', 'svr_poly', 'cv')
y_pos = np.arange(len(objects))
performance = [LRM_r2_test, svr_lin_r2_test, svr_rbf_r2_test, svr_poly_r2_test, cv_r2_test]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=45)
plt.ylabel('R2')
plt.xlabel('Model')
plt.title('R2 test set')


fig.tight_layout()
plt.show()

