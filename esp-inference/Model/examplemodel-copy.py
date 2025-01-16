import tensorflow as tf
import numpy as np

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(1,3), unroll=True, return_sequences=False),  # Input shape: 10 timesteps, 1 feature
    tf.keras.layers.Dense(3, activation='linear')               # Output: single value
])

model.compile(optimizer='adam', loss='mse')

# Generate random training data
# X_train: 1000 samples, each with 10 timesteps and 1 feature
# y_train: corresponding target values
X_train = np.random.rand(1000,1, 3).astype(np.float32)  # Random inputs
y_train = np.random.rand(1000, 3).astype(np.float32)      # Random targets

# Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("model1.tflite", "wb") as f:
    f.write(tflite_model)

print("Model trained and converted to TensorFlow Lite!")
