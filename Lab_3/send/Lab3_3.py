#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def search_parameter(model):
    p = {
        # 'activation':['relu', 'sigmoid', 'tanh', 'softmax'],
         'optimizer': ['Adam', 'adadelta'],
         'losses': ['categorical_crossentropy'],
         'batch_size': [8, 16, 20, 40, 60, 80, 100],
         'epochs': [10, 50, 100]
         }

    def cnn_model(X_train, y_train, x_val, y_val, params):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:]), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(128 , activation='relu' ))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        out = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], validation_data=[x_val, y_val], verbose=0)
        return out, model

    def lstm_model(X_train, y_train, x_val, y_val, params):
        model = Sequential()
        model.add(LSTM(32, input_shape=(X_train.shape[1:]) ))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
        optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        out = model.fit( X_train, y_train, batch_size=params['batch_size'], validation_data=[x_val, y_val], epochs=params['epochs'], verbose=0)
        return out, model

    if model == 'cnn' :
      scan_object = ta.Scan(X_train, y_train, model=cnn_model, params=p, experiment_name='cnn', fraction_limit=0.1, x_val=X_valid, y_val=y_valid)
    else :
      scan_object = ta.Scan(X_2d_train, y_train, model=lstm_model, params=p, experiment_name='lstm', fraction_limit=0.1, x_val=X_2d_valid, y_val=y_valid)

    return scan_object

def cnn_model(optimizer, losses) :
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
  model.compile( loss=losses, optimizer=optimizer, metrics=['accuracy'])
  return model

def lstm_model(optimizer, losses) :
  model = Sequential()
  model.add(LSTM(32, input_shape=(X_2d_train.shape[1:]) ))
  model.add(Dense(y_train.shape[1], activation='sigmoid'))
  model.compile( loss=losses, optimizer=optimizer, metrics=['accuracy'])
  return model

def show_search_res(scan, model) :
  # use Scan object as input
  analyze_object = ta.Analyze(scan)
  print(analyze_object.data)
  print('=================================================')

  # get the highest result for any metric
  print('Low validate loss : ', analyze_object.high('val_accuracy'))
  print('=================================================')

  # get the round with the best result
  print('\nindex of best result :', analyze_object.rounds2high('val_accuracy'))
  print('=================================================')

  # evaluae with k fold
  e = ta.Evaluate(scan)
  model_evaluate = list()
  if model == 'lstm' :
    xx = X_2d_train
    yy = y_train
  else :
    xx = X_train
    yy = y_train
  for i in range(len(analyze_object.data)) :
    evaluate = e.evaluate(xx, yy, folds=10, metric='val_accuracy', task='multi_label', model_id=i)
    model_evaluate.append(np.array(evaluate).mean())
  print('evaluate with kfold', model_evaluate)
  print('=================================================')

  # get the best paramaters
  print('\nbest parameters :')
  print(analyze_object.best_params('val_accuracy', ['acc', 'loss', 'val_loss']))
  analyze_object.plot_line('val_accuracy')
  return analyze_object, model_evaluate


scan = search_parameter('cnn')
print(scan.details)
print('=================================================')
analyze_object_cnn, model_cnn_evaluate = show_search_res(scan, 'cnn')

scan = search_parameter('lstm')
print(scan.details)
print('=================================================')
analyze_object_lstm, model_lstm_evaluate = show_search_res(scan, 'lstm')

cnn_best_params = analyze_object_cnn.data.iloc[analyze_object_cnn.rounds2high('val_accuracy'), :]
print('CNN best score : ', model_cnn_evaluate[analyze_object_cnn.rounds2high('val_accuracy')])
print('CNN best parameters : \n', cnn_best_params)

print('==================================')

lstm_best_params = analyze_object_lstm.data.iloc[analyze_object_lstm.rounds2high('val_accuracy'), :]
print('LSTM best score : ', model_lstm_evaluate[analyze_object_lstm.rounds2high('val_accuracy')])
print('LSTM best parameters : \n', lstm_best_params)


# save
cnn_best_params.to_csv(root_dir + '/cnn_best_params.csv')
lstm_best_params.to_csv(root_dir + '/lstm_best_params.csv')

model = cnn_model(cnn_best_params['optimizer'], cnn_best_params['losses'])
model.summary()
history = model.fit(X_train, y_train, batch_size=cnn_best_params['batch_size'], validation_data=(X_valid, y_valid), epochs=cnn_best_params['epochs'], verbose=1)

# ใช้โมเดลที่สอนจากพารามิเตอร์ที่ดีที่สุดมา predict ข้อมูล ชุด x_test
y_prediction = model.predict(X_test)
y_pred_single = [np.argmax(p) for p in y_prediction]
y_test_single=[np.argmax(p) for p in y_test]

# คำนวณค่าตัววัดประสิทธิภาพของการทำนายจากโมเดล CNN

print(classification_report(y_test_single, y_pred_single))

conf_mat = confusion_matrix(y_test_single, y_pred_single)
plt.figure(figsize = (10, 7))
ax = sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels='0 1 2 3 4'.split(), yticklabels='0 1 2 3 4'.split(), cmap="Blues")
bottom, top = ax.get_ylim()
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


model = lstm_model(lstm_best_params['optimizer'], lstm_best_params['losses'])
model.summary()
history = model.fit(X_2d_train, y_train, batch_size=lstm_best_params['batch_size'], validation_data=(X_2d_valid, y_valid), epochs=lstm_best_params['epochs'], verbose=1)

y_prediction = model.predict(X_2d_test)
y_pred_single = [np.argmax(p) for p in y_prediction]
y_test_single = [np.argmax(p) for p in y_test]

# คำนวณค่าตัววัดประสิทธิภาพของการทำนายจากโมเดล CNN

print(classification_report(y_test_single, y_pred_single))

conf_mat = confusion_matrix(y_test_single, y_pred_single)
plt.figure(figsize = (10, 7))
ax = sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels='0 1 2 3 4'.split(), yticklabels='0 1 2 3 4'.split(), cmap="Blues")
bottom, top = ax.get_ylim()
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

