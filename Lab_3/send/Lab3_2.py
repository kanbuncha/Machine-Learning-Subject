#!/usr/bin/env python
# coding: utf-8

# In[18]:


# CNN
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:]), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128 , activation='relu' ))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
model.summary()

# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, validation_data=(X_valid, y_valid), epochs=100, verbose=1)

# predict CNN
y_prediction = model.predict(X_test)
y_pred_single = [np.argmax(p) for p in y_prediction]
y_test_single=[np.argmax(p) for p in y_test]

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

acc = history.history['accuracy']
loss = history.history['loss']
ax1.plot(acc, label=model)
ax2.plot(loss, label=model)
    
ax1.set_ylabel('Training accuracy')
ax2.set_ylabel('Training loss')
ax2.set_xlabel('epochs')
plt.show()

# คำนวณค่าตัววัดประสิทธิภาพของการทำนายจากโมเดล CNN 
print(classification_report(y_test_single, y_pred_single))

conf_mat = confusion_matrix(y_test_single, y_pred_single)
plt.figure(figsize = (10, 7))
ax = sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels='0 1 2 3 4'.split(), yticklabels='0 1 2 3 4'.split(), cmap="Blues")
bottom, top = ax.get_ylim()
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


# LSTM

model = Sequential()
model.add(LSTM(32, input_shape=(X_2d_train.shape[1:]) ))


# model.add(Dropout(0.25))

# model.add(Dropout(0.25))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
model.summary()

optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit( X_2d_train, y_train, batch_size=16, validation_data=(X_2d_valid, y_valid), epochs=100)

# predict LSTM
y_prediction = model.predict(X_2d_test)
y_pred_single = [np.argmax(p) for p in y_prediction]
y_test_single = [np.argmax(p) for p in y_test]

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

acc = history.history['accuracy']
loss = history.history['loss']
ax1.plot(acc, label=model)
ax2.plot(loss, label=model)
    
ax1.set_ylabel('Training accuracy')
ax2.set_ylabel('Training loss')
ax2.set_xlabel('epochs')
#ax1.legend()
#ax2.legend()
plt.show()

# คำนวณค่าตัววัดประสิทธิภาพของการทำนายจากโมเดล LSTM
print(classification_report(y_test_single, y_pred_single))

conf_mat = confusion_matrix(y_test_single, y_pred_single)
plt.figure(figsize = (10, 7))
ax = sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels='0 1 2 3 4'.split(), yticklabels='0 1 2 3 4'.split(), cmap="Blues")
bottom, top = ax.get_ylim()
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

