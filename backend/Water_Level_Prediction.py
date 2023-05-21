# -*- coding: utf-8 -*-
"""Water_Level_Prediction.ipynb
---WATER LEVEL PREDICITION ---
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

"""Load data"""

data = pd.read_csv('waterlevels.csv')
X = data.drop(['Water Level'], axis=1)
y = data['Water Level'].values.reshape(-1, 1)
evap = data['Evaporation'].values.reshape(-1, 1)
salinity = data['Salinity'].values.reshape(-1,1 )
precipitation = data['Precipitation'].values.reshape(-1, 1)

"""Checking data"""

data.head()

"""Normalize data"""

min_max_list=[]
temp=[]

# for i in data.columns:
#     print(df[i].max())
#     print(df[i].min())
    
for i in data.columns:
    temp.append(data[i].max())
    temp.append(data[i].min())
    min_max_list.append(temp)
    temp=[]

min_max_list

"""Combine X and evap, salinity, and precipitation data"""

scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

scaler_evap = MinMaxScaler()
evap = scaler_evap.fit_transform(evap)

scaler_salinity = MinMaxScaler()
salinity = scaler_salinity.fit_transform(salinity)

scaler_precipitation = MinMaxScaler()
precipitation = scaler_precipitation.fit_transform(precipitation)

"""another way to Combine X and evap, salinity, and precipitation data"""

# X = np.concatenate((X, evap, salinity, precipitation), axis=1)

"""Split data into training and testing sets"""

split = int(0.9* len(data))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

"""Define MLP model with ReLU activation"""

model = Sequential()
model.add(Dense(64, input_dim=(X.shape[1]), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1,  activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

"""Train model"""

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

"""Evaluate model on training set"""

train_score = model.evaluate(X_train, y_train, verbose=0)
print('Training loss:', train_score)

"""Evaluate model on testing set"""

test_score = model.evaluate(X_test, y_test, verbose=0)
print('Testing loss:', test_score)

X_test

"""Make predictions"""

y_test *= 1000
y_test += 35
predictions = scaler_y.inverse_transform(model.predict(X_test))

"""Plot predictions against actual values"""

import matplotlib.pyplot as plt
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlim(0,22)
plt.ylim(0,500)
plt.ylabel("Water Level(m)")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()

type(X_test[0])
X_test[0]

"""Plot loss curves for training and testing sets"""

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.xlabel("No. of Epoch")
plt.title("Training vs Testing")
plt.legend()
plt.show()

model.save('model.h5')
df.head()
