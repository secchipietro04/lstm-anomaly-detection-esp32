import tensorflow as tf
import numpy as np

# === Model Parameters === #
SEQUENCE_LENGTH = 1024
FEATURE_SIZE = 3
LSTM_UNITS_1 = 16
LSTM_UNITS_2 = 8

# === Training Model === #
training_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
    tf.keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True, name='lstm_1'),
    tf.keras.layers.LSTM(LSTM_UNITS_2, return_sequences=True, name='lstm_2'),
    tf.keras.layers.Dense(FEATURE_SIZE, activation='linear')
])

# Compile and Train the Model
training_model.compile(optimizer='adam', loss='mse')
X_train = np.random.rand(10, SEQUENCE_LENGTH, FEATURE_SIZE).astype(np.float32)
y_train = np.random.rand(10, SEQUENCE_LENGTH, FEATURE_SIZE).astype(np.float32)
training_model.fit(X_train, y_train, epochs=2, batch_size=2)

# === Inference Model with Flattened Input/Output === #

# Flattened input size: input data + hidden and cell states
flattened_input_size = FEATURE_SIZE + (LSTM_UNITS_1 * 2) + (LSTM_UNITS_2 * 2)

# Flattened input layer
flattened_input = tf.keras.Input(shape=(flattened_input_size,), name='Flattened_Input')

# === Use tf.slice to split the input === #

# Input Data
input_data = tf.keras.layers.Lambda(lambda x: tf.slice(x, [0, 0], [-1, FEATURE_SIZE]), name='Slice_Input')(flattened_input)

# Hidden State and Cell State for LSTM 1
hidden_state_1 = tf.keras.layers.Lambda(lambda x: tf.slice(x, [0, FEATURE_SIZE], [-1, LSTM_UNITS_1]), name='Slice_H1')(flattened_input)
cell_state_1 = tf.keras.layers.Lambda(lambda x: tf.slice(x, [0, FEATURE_SIZE + LSTM_UNITS_1], [-1, LSTM_UNITS_1]), name='Slice_C1')(flattened_input)

# Hidden State and Cell State for LSTM 2
hidden_state_2 = tf.keras.layers.Lambda(lambda x: tf.slice(x, [0, FEATURE_SIZE + LSTM_UNITS_1 * 2], [-1, LSTM_UNITS_2]), name='Slice_H2')(flattened_input)
cell_state_2 = tf.keras.layers.Lambda(lambda x: tf.slice(x, [0, FEATURE_SIZE + LSTM_UNITS_1 * 2 + LSTM_UNITS_2], [-1, LSTM_UNITS_2]), name='Slice_C2')(flattened_input)

# === LSTM Cells === #
lstm_cell_1 = tf.keras.layers.LSTMCell(LSTM_UNITS_1, name="LSTM_Cell_1")
lstm_cell_2 = tf.keras.layers.LSTMCell(LSTM_UNITS_2, name="LSTM_Cell_2")

# First LSTM Layer
lstm_output_1, [new_hidden_state_1, new_cell_state_1] = lstm_cell_1(
    input_data, states=[hidden_state_1, cell_state_1]
)

# Second LSTM Layer
lstm_output_2, [new_hidden_state_2, new_cell_state_2] = lstm_cell_2(
    lstm_output_1, states=[hidden_state_2, cell_state_2]
)

# Dense layer to generate final output
final_output = tf.keras.layers.Dense(FEATURE_SIZE, activation='linear', name='Final_Output')(lstm_output_2)

# === Concatenate Final Output and Updated States === #
flattened_output = tf.keras.layers.Concatenate(name="Flattened_Output")([
    final_output,
    new_hidden_state_1, new_cell_state_1,
    new_hidden_state_2, new_cell_state_2
])

# === Define the Model === #
flattened_model = tf.keras.Model(inputs=flattened_input, outputs=flattened_output, name="Flattened_LSTM_Model")

# === Save the Model and Convert to TFLite === #
flattened_model.save("flattened_lstm_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(flattened_model)
tflite_model = converter.convert()

# Save the TFLite model
with open("flattened_lstm_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Flattened LSTM model saved and converted to TFLite!")

# === Test the Model === #
test_input = np.random.rand(1, flattened_input_size).astype(np.float32)
output = flattened_model.predict(test_input)
print("Model Output:\n", output)
