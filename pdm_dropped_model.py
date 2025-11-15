import numpy as np
import pandas as pd 
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


data = pd.read_csv("ai 2020.csv")
data.drop(columns=['UDI','Product ID'], inplace=True)

le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])

X = data.drop(columns=['Machine failure', 'Process temperature [K]', 'Tool wear [min]', 'Torque [Nm]', 'TWF', 'HDF'])
y = data['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')


#rf = RandomForestClassifier(random_state=42)
#rf.fit(X_train, y_train)
#y_pred = rf.predict(X_test)

#print(classification_report(y_test, y_pred))
#print('ROC AUC Score:', roc_auc_score(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))

model.save("saved_pdm_dropped_model.keras")
