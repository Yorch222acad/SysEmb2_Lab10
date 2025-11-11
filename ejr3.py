import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('spam_text_data.csv')
print(df.head())

X = df['Message']
y = df['Category'].map({'spam': 1, 'ham': 0})

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(X)

model = keras.Sequential()

model.add(keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(layers.Embedding(input_dim=max_features + 1, output_dim=16))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X.to_numpy(), y, epochs=20, validation_split=0.2, batch_size=32)

loss, accuracy = model.evaluate(X.to_numpy(), y)
print("\nPrecisión (Accuracy) final en los datos de entrenamiento: {:.2f}%".format(accuracy * 100))

# Ejemplo 1 (Spam)
pred_spam = model.predict(tf.constant(["gana dinero rapido haz click aqui premio"]))
print(f"Predicción (Spam): {pred_spam[0][0]:.4f}")

# Ejemplo 2 (No Spam)
pred_no_spam = model.predict(tf.constant(["hola juan, nos vemos en la reunion de mañana"]))
print(f"Predicción (No Spam): {pred_no_spam[0][0]:.4f}")