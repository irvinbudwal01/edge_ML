import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv('air_quality_dataset.csv')

df['DATEOFF'] = pd.to_datetime(df['DATEOFF'], errors='coerce')
df['DATEON'] = pd.to_datetime(df['DATEON'], errors='coerce')

#dropping a few additional columns past 'dateon'

numeric_columns = df.columns.drop(['SITE_ID', 'DATEOFF', 'DATEON', 'Week', 'Year', 'SO4', 'HNO3', 'K', 'NH4', 'SO2']).tolist()

#print(numeric_columns)

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

#https://www.kaggle.com/code/atifmasih/air-qaulity-categorization-using-randomforest-94

means = df.select_dtypes(include=['int64','float64']).mean()

df = df.fillna(means)

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numeric_columns])

df['Air_Quality_Index'] = np.mean(scaled_features, axis=1)

df['Air_Quality_Category'] = pd.cut(df['Air_Quality_Index'], bins=[-np.inf, -1, 0, 1, np.inf], labels=['Very Poor', 'Poor', 'Moderate', 'Good'])

X = df[numeric_columns]
y = df['Air_Quality_Category']

from tensorflow.keras.utils import to_categorical

y = df['Air_Quality_Category'].cat.codes.astype('int32')

X = scaled_features.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the model
#model = RandomForestClassifier(n_estimators=100, random_state=42)
#model.fit(X_train, y_train)

# Make predictions
#y_pred = model.predict(X_test)

# Evaluate the model
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:\n", classification_report(y_test, y_pred))

model = keras.Sequential([keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(4, activation="softmax")])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print(f'\nTest accuracy: {test_acc}')

model.save("saved_env_dropped_model.keras")

#loaded_model = tf.keras.models.load_model("saved_env_model.keras")

#predictions = loaded_model.predict(X_test, batch_size=32, verbose=1)

