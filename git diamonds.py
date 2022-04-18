# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 00:43:32 2022

@author: User
"""
#https://www.kaggle.com/datasets/shivam2503/diamonds
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

data = pd.read_csv(r"X:\Users\User\Tensorflow Deep Learning\csv\diamonds.csv", header=0)

#%%
data = data.drop(data.columns[0], axis=1)
data = data.drop(data[data['x'] == 0].index)
data = data.drop(data[data['y'] == 0].index)
data = data.drop(data[data['z'] == 0].index)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_list = ['carat', 'x', 'y', 'z', 'depth', 'table']
data[ss_list] = ss.fit_transform(data[ss_list])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_list = ['cut', 'color', 'clarity']
data[le_list] = data[le_list].apply(le.fit_transform)

#%%
feature = data.copy()
label = feature.pop('price')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=12345)

#%%
inputs = x_train.shape[-1]
output = y_train.shape[-1]

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=inputs))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

#%%
import datetime as dt
log_path = r"X:\Users\User\Tensorflow Deep Learning\Tensorboard\diamondsprice" + dt.datetime.now().strftime("%H%M%S")
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=100, callbacks=[es_callback, tb_callback])

#%%
predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
save_path = r"X:\Users\User\Tensorflow Deep Learning\github\graphs image"
plt.savefig(os.path.join(save_path, "diamondsprice.png"), bbox_inches='tight')
plt.show()