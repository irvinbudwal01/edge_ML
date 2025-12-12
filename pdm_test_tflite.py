import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("ai 2020.csv")

data.drop(columns=['UDI', 'Product ID'], inplace=True)

le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])

#drop everything after machine failure for dropped model params
#X = data.drop(columns=['Machine failure', 'Process temperature [K]', 'Tool wear [min]', 'Torque [Nm]', 'TWF', 'HDF'])
X = data.drop(columns=['Machine failure'])
y = data['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

interpreter = tf.lite.Interpreter(model_path="pdm_model.tflite")

#interpreter = tf.lite.Interpreter(model_path="pdm_dropped_model.tflite")

#interpreter = tf.lite.Interpreter(model_path="pdm_student_model.tflite")

#interpreter = tf.lite.Interpreter(model_path="pdm_student_dropped_model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

np_features = np.array(X_test, dtype=np.float32)

inp_idx, out_idx = input_details["index"], output_details["index"]

preds = []

for i in range(np_features.shape[0]):
    x1 = np_features[i:i+1]
    interpreter.set_tensor(inp_idx, x1)
    interpreter.invoke()
    y = interpreter.get_tensor(out_idx)
    predicted_class = np.argmax(y)
    preds.append(predicted_class)

accuracy = accuracy_score(y_test, preds)
print(f"TFLite Model Acc: {accuracy:.4f}")



