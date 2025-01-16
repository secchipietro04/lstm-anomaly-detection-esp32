import tensorflow as tf
import numpy as np

# Training model parameters
SEQUENCE_LENGTH = 1024
FEATURE_SIZE = 3
LSTM_UNITS_1 = 16  # First LSTM layer units
LSTM_UNITS_2 = 8   # Second LSTM layer units

# Define the training model
training_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
    tf.keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True, name='lstm_1'),
    tf.keras.layers.LSTM(LSTM_UNITS_2, return_sequences=True, name='lstm_2'),
    tf.keras.layers.Dense(FEATURE_SIZE, activation='linear')
])

# Train the model
training_model.compile(optimizer='adam', loss='mse')
X_train = np.random.rand(10, SEQUENCE_LENGTH, FEATURE_SIZE).astype(np.float32)
y_train = np.random.rand(10, SEQUENCE_LENGTH, FEATURE_SIZE).astype(np.float32)
training_model.fit(X_train, y_train, epochs=2, batch_size=2)

# Create a single-step inference model using LSTMCell
input_step = tf.keras.Input(shape=(1, FEATURE_SIZE), name='A_input_step')

# States for first LSTM layer
hidden_state_input_1 = tf.keras.Input(shape=(LSTM_UNITS_1,), name='B_hidden_state_input_1')
cell_state_input_1 = tf.keras.Input(shape=(LSTM_UNITS_1,), name='C_cell_state_input_1')

# States for second LSTM layer
hidden_state_input_2 = tf.keras.Input(shape=(LSTM_UNITS_2,), name='D_hidden_state_input_2')
cell_state_input_2 = tf.keras.Input(shape=(LSTM_UNITS_2,), name='E_cell_state_input_2')

# Create LSTMCells and manage states manually
lstm_cell_1 = tf.keras.layers.LSTMCell(LSTM_UNITS_1, name="A_lstm_cell_1")
lstm_cell_2 = tf.keras.layers.LSTMCell(LSTM_UNITS_2, name="B_lstm_cell_2")

# First LSTM layer
lstm_output_1, [hidden_state_1, cell_state_1] = lstm_cell_1(
    input_step[:, 0, :], states=[hidden_state_input_1, cell_state_input_1]
)

# Second LSTM layer
lstm_output_2, [hidden_state_2, cell_state_2] = lstm_cell_2(
    lstm_output_1, states=[hidden_state_input_2, cell_state_input_2]
)

# Define a custom layer to apply the Dense operation
class ApplyDenseLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ApplyDenseLayer, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(FEATURE_SIZE, activation='linear')

    def call(self, inputs):
        return self.dense_layer(inputs)

# Apply the dense layer to the second LSTM output
output_step = ApplyDenseLayer()(lstm_output_2)

# Build single-step model with states from both LSTM layers
single_step_model = tf.keras.Model(
    inputs=[
        input_step, 
        hidden_state_input_1, cell_state_input_1,
        hidden_state_input_2, cell_state_input_2
    ],
    outputs=[
        output_step, 
        hidden_state_1, cell_state_1,
        hidden_state_2, cell_state_2
    ]
)

# Naming the outputs explicitly
output_step.name = 'A_output_step'
hidden_state_1.name = 'B_hidden_state_1'
cell_state_1.name = 'C_cell_state_1'
hidden_state_2.name = 'D_hidden_state_2'
cell_state_2.name = 'E_cell_state_2'

# Save and convert to TFLite
single_step_model.save("model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(single_step_model)

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Single-step model with two LSTM layers saved and converted to TFLite!")
