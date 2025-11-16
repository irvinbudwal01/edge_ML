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
#keras_model = tf.keras.models.load_model("saved_env_model.keras")

keras_model = tf.keras.models.load_model("saved_pdm_model.keras")

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
            
            distillation_loss = (self.distillation_loss_fn(tf.nn.sigmoid(teacher_predictions / self.temperature), tf.nn.sigmoid(student_predictions / self.temperature),))
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

data = pd.read_csv("ai 2020.csv")
data.drop(columns=['UDI','Product ID'],inplace=True)

le = LabelEncoder()
data['Type'] = le.fit_transform(data['Type'])

X = data.drop(columns=['Machine failure'])
y = data['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

student = keras.Sequential([keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")])

student_scratch = keras.models.clone_model(student)

distiller = Distiller(student=student, teacher=teacher)

distiller.compile(optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.BinaryAccuracy()],
        student_loss_fn=keras.losses.BinaryCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=.1,
        temperature=10,)

distiller.fit(X_train, y_train, epochs=20)

student.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()],
)
student.evaluate(X_test, y_test)

teacher.evaluate(X_test, y_test)

student.save("saved_pdm_student_model.keras")

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
