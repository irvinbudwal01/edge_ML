import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
#keras_model = tf.keras.models.load_model("saved_env_model.keras")

keras_model = tf.keras.models.load_model("saved_env_model.keras")

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=.1, temperature=3):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            
            distillation_loss = (self.distillation_loss_fn(tf.nn.softmax(teacher_predictions / self.temperature, axis=1), tf.nn.softmax(student_predictions / self.temperature, axis=1),))
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss}
                )
        return results

    def test_step(self, data):
        x, y = data

        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)
        self.compiled_metrics.update_state(y, y_prediction)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

teacher = keras_model

#student = keras.Sequential([keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
#    keras.layers.Dense(16, activation="relu"),
#    keras.layers.Dense(1, activation="sigmoid")])

#student_scratch = keras.models.clone_model(student)

df = pd.read_csv('air_quality_dataset.csv')

df['DATEOFF'] = pd.to_datetime(df['DATEOFF'], errors='coerce')
df['DATEON'] = pd.to_datetime(df['DATEON'], errors='coerce')

numeric_columns = df.columns.drop(['SITE_ID', 'DATEOFF', 'DATEON']).tolist()

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

student = keras.Sequential([keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)), keras.layers.Dense(32, activation="relu"), keras.layers.Dense(4, activation="softmax")])

student_scratch = keras.models.clone_model(student)

distiller = Distiller(student=student, teacher=teacher)

distiller.compile(optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=.1,
        temperature=10,)

distiller.fit(X_train, y_train, epochs=20)

student.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
student.evaluate(X_test, y_test)

teacher.evaluate(X_test, y_test)

student.save("saved_env_student_model.keras")

#keras_model = tf.keras.models.load_model("saved_env_dropped_model.keras")

#converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8  # Specify input and output types for integer quantization
#converter.inference_output_type = tf.int8

#tf_lite_model = converter.convert()

#with open('env_model.tflite', 'wb') as f:
#    f.write(tf_lite_model)

#with open('env_dropped_model.tflite', 'wb') as f:
#    f.write(tf_lite_model)
