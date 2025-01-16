import tensorflow as tf
import numpy as np

def build_flattened_lstm_model_from_layers(feature_size, lstm_layers):
    """
    Builds a step-by-step LSTM model using trained LSTM layers.
    
    Args:
        feature_size (int): Number of input features.
        lstm_layers (list): List of trained tf.keras.layers.LSTM layers.

    Returns:
        model (tf.keras.Model): Step-by-step LSTM model.
        flattened_input_size (int): Size of the flattened input vector.
    """
    # Extract the number of units in each LSTM layer
    lstm_units_list = [layer.units for layer in lstm_layers]

    # Total input size = feature_size + 2 * sum of LSTM units (hidden + cell states)
    flattened_input_size = feature_size + 2 * sum(lstm_units_list)

    # Input layer
    flattened_input = tf.keras.Input(shape=(flattened_input_size,), name='Flattened_Input')

    # === Manually slice the input using tf.slice === #
    current_index = 0

    # Slice input data
    input_data = tf.keras.layers.Lambda(
        lambda x: tf.slice(x, [0, current_index], [-1, feature_size]),
        name='Slice_Input'
    )(flattened_input)
    current_index += feature_size

    # Slice hidden and cell states
    hidden_states = []
    cell_states = []

    for i, units in enumerate(lstm_units_list):
        # Calculate the indices explicitly for each slice
        start_index = sum(lstm_units_list[:i])  # Start index for the current slice
        end_index = start_index + units         # End index for the current slice

        print(f"0, {start_index}, -1, {units}")

        # Slice hidden state
        h_state = tf.keras.layers.Lambda(
            lambda x, start=start_index, u=units: tf.slice(x, [0, start], [-1, u]),
            name=f'Slice_Hidden_{i+1}'
        )(flattened_input)

        print(f"0, {end_index}, -1, {units}")

        # Slice cell state
        c_state = tf.keras.layers.Lambda(
            lambda x, start=end_index, u=units: tf.slice(x, [0, start], [-1, u]),
            name=f'Slice_Cell_{i+1}'
        )(flattened_input)

        hidden_states.append(h_state)
        cell_states.append(c_state)
    # === Convert LSTM layers to LSTMCell === #
    lstm_cells = [tf.keras.layers.LSTMCell(units=layer.units, name=f"LSTM_Cell_{i+1}")
                  for i, layer in enumerate(lstm_layers)]

    # === Pass through LSTM cells === #
    lstm_output = input_data
    updated_hidden_states = []
    updated_cell_states = []

    for i, lstm_cell in enumerate(lstm_cells):
        lstm_output, [new_hidden, new_cell] = lstm_cell(lstm_output, states=[hidden_states[i], cell_states[i]])
        updated_hidden_states.append(new_hidden)
        updated_cell_states.append(new_cell)

    # === Final Dense Layer === #
    final_output = tf.keras.layers.Dense(feature_size, activation='linear', name='Final_Output')(lstm_output)

    # === Concatenate final output and updated states using tf.concat === #
    concatenated_output = tf.keras.layers.Lambda(
        lambda tensors: tf.concat(tensors, axis=-1),
        name='Concatenate_Output'
    )([final_output] + updated_hidden_states + updated_cell_states)

    # === Build the model === #
    model = tf.keras.Model(inputs=flattened_input, outputs=concatenated_output, name="Step_LSTM_Model")

    return model, flattened_input_size

# === Step 1: Train a Sequential LSTM Model === #
SEQUENCE_LENGTH = 10
FEATURE_SIZE = 3
LSTM_UNITS_LIST = [16, 8]  # Two LSTM layers

# Define the training model
training_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
    tf.keras.layers.LSTM(LSTM_UNITS_LIST[0], return_sequences=True, name='LSTM_1'),
    tf.keras.layers.LSTM(LSTM_UNITS_LIST[1], return_sequences=False, name='LSTM_2'),
    tf.keras.layers.Dense(FEATURE_SIZE, activation='linear', name='Output')
])

# Compile the training model
training_model.compile(optimizer='adam', loss='mse')

# Dummy data for training
X_train = np.random.rand(100, SEQUENCE_LENGTH, FEATURE_SIZE).astype(np.float32)
y_train = np.random.rand(100, FEATURE_SIZE).astype(np.float32)

# Train the model
training_model.fit(X_train, y_train, epochs=2, batch_size=8)

# === Step 2: Extract the Trained LSTM Layers === #
trained_lstm_layers = [layer for layer in training_model.layers if isinstance(layer, tf.keras.layers.LSTM)]

# === Step 3: Build the Flattened LSTM Model === #
flattened_model, input_size = build_flattened_lstm_model_from_layers(FEATURE_SIZE, trained_lstm_layers)

# Model summary and input size
flattened_model.summary()
print(f"Flattened Input Size: {input_size}")

# === Step 4: Convert to TFLite === #
flattened_model.save("step_lstm_model_with_slice.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(flattened_model)
tflite_model = converter.convert()

# Save TFLite model
with open("step_lstm_model_with_slice.tflite", "wb") as f:
    f.write(tflite_model)

print("Step-by-step LSTM model saved and converted to TFLite!")

# === Step 5: Test the Flattened Model === #
test_input = np.random.rand(1, input_size).astype(np.float32)
output = flattened_model.predict(test_input)
print("Model Output:\n", output)
