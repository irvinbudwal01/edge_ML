import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import tflite_runtime.interpreter as tf_lite_interpreter

#extract features

df = pd.read_csv('air_quality_dataset.csv')

df['DATEOFF'] = pd.to_datetime(df['DATEOFF'], errors='coerce')
df['DATEON'] = pd.to_datetime(df['DATEON'], errors='coerce')

numeric_columns = df.columns.drop(['SITE_ID', 'DATEOFF', 'DATEON']).tolist()

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

means = df.select_dtypes(include=['int64','float64']).mean()

df= df.fillna(means)

numeric_columns = df.select_dtypes(include=['float64','int64']).columns

scaler = StandardScaler()

scaled_features = scaler.fit_transform(df[numeric_columns])

df['Air_Quality_Index'] = np.mean(scaled_features, axis=1)

df['Air_Quality_Category'] = pd.cut(df['Air_Quality_Index'], bins=[-np.inf,-1,0,1,np.inf], labels=['Very Poor', 'Poor', 'Moderate', 'Good'])

X = df[numeric_columns]
y= df['Air_Quality_Category']

from tensorflow.keras.utils import to_categorical

y = df['Air_Quality_Category'].cat.codes.astype('int32')

X = scaled_features.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

interpreter = tf.lite.Interpreter(model_path="env_model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#print(output_details)

#input_shape = input_details[0]['shape']

np_features = np.array(X_test, dtype=np.float32)
#print(np_features)
np_features = np.expand_dims(np_features, axis=0)
#print(np_features)
interpreter.set_tensor(input_details[0]['index'], np_features)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

print("Inference Output is {}".format(output_data))

