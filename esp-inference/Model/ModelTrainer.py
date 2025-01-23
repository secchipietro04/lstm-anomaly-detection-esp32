import tensorflow as tf
import numpy as np
from datetime import timedelta, datetime
from mockseries.trend import FlatTrend
from mockseries.seasonality import SinusoidalSeasonality
from mockseries.noise import RedNoise
from mockseries.utils import datetime_range, plot_timeseries
import code
import random
import warnings

import matplotlib.pyplot as plt


class LSTMModelTrainer:
    def __init__(self, feature_size=3, encoder_units=None, decoder_units=None, sequence_length=40, model=None, dataset=None):
        # Use default values if None is provided for encoder_units and decoder_units
        self.encoder_units = encoder_units if encoder_units is not None else [
            16, 8]
        self.decoder_units = decoder_units if decoder_units is not None else [
            8, 16]
        
        self.feature_size = feature_size
        self.sequence_length = sequence_length

        # Initialize model if not provided
        self.model = model if model is not None else self.build_autoencoder()

        # Initialize dataset as an empty numpy array if not provided
        self.dataset = dataset if dataset is not None else []

    def chunk_time_series( data,chunk_size):

        n_series, length, features = data.shape
        merged_chunks = []
        step_size = chunk_size // 2  # Overlap by half

        for series in range(n_series):
            for start in range(0, length - chunk_size + 1, step_size):
                end = start + chunk_size
                chunk = data[series, start:end, :]  # Extract chunk
                merged_chunks.append(chunk)

        return np.array(merged_chunks)

    def __train(self, model, data, epochs=50, batch_size=32, validation_split=0.1):
        if data.size == 0:
            raise ValueError(
                "Dataset is empty. Please provide valid training data.")

        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        )

        # Add learning rate reduction on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        history = model.fit(
            data,
            data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return history

    def train_autoencoder(self, epochs=50, batch_size=32, validation_split=0.1):
        """
        Public method to train the autoencoder using the internal dataset.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data used for validation.
        """
        if self.dataset.size == 0:
            print("Dataset is empty. Generating synthetic data...")
            self.dataset = LSTMModelTrainer.chunk_time_series(
                 self.generate_timeseries(n_points=self.sequence_length*10), self.sequence_length)

        print("Starting model training...")
        history = self.__train(self.model, self.dataset,
                               epochs, batch_size, validation_split)
        print("Training completed.")
        return history

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
        flattened_input = tf.keras.Input(
            shape=(flattened_input_size,), name='Flattened_Input')

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
            # Start index for the current slice
            start_index = sum(lstm_units_list[:i])
            end_index = start_index + units         # End index for the current slice

            print(f"0, {start_index}, -1, {units}")

            # Slice hidden state
            h_state = tf.keras.layers.Lambda(
                lambda x, start=start_index, u=units: tf.slice(
                    x, [0, start], [-1, u]),
                name=f'Slice_Hidden_{i+1}'
            )(flattened_input)

            print(f"0, {end_index}, -1, {units}")

            # Slice cell state
            c_state = tf.keras.layers.Lambda(
                lambda x, start=end_index, u=units: tf.slice(
                    x, [0, start], [-1, u]),
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
            lstm_output, [new_hidden, new_cell] = lstm_cell(
                lstm_output, states=[hidden_states[i], cell_states[i]])
            updated_hidden_states.append(new_hidden)
            updated_cell_states.append(new_cell)

        # === Final Dense Layer === #
        final_output = tf.keras.layers.Dense(
            feature_size, activation='linear', name='Final_Output')(lstm_output)

        # === Concatenate final output and updated states using tf.concat === #
        concatenated_output = tf.keras.layers.Lambda(
            lambda tensors: tf.concat(tensors, axis=-1),
            name='Concatenate_Output'
        )([final_output] + updated_hidden_states + updated_cell_states)

        # === Build the model === #
        model = tf.keras.Model(
            inputs=flattened_input, outputs=concatenated_output, name="Step_LSTM_Model")

        return model, flattened_input_size

    def export_model_to_cc_from_file(input_file, output_file, model_name):
        with open(input_file, "rb") as f:
            model_content = f.read()

        LSTMModelTrainer.export_model_to_cc(
            model_content, output_file, model_name)

    def export_model_to_tflite(model):
        if isinstance(model, tf.keras.Model):
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                tflite_model = converter.convert()
                return tflite_model
            except Exception as e:
                warnings.warn(
                    f"An error occurred during model conversion: {e}")
                return None
        else:
            warnings.warn(
                "The provided object is not a tf.keras.Model instance. Conversion skipped.")
            return model

    def export_model_to_tflite_file(model, filename="model.tflite"):
        model = LSTMModelTrainer.export_model_to_tflite(model)
        if model is not None:
            with open(filename, "wb") as f:
                f.write(model)
            print(f"Model saved as {filename}")

    def export_model_to_cc(model_content, output_file, model_name):
        model_content = LSTMModelTrainer.convert_to_tflite(model_content)
        with open(output_file, "w") as f:
            f.write('#include "model.h"\n\n')
            f.write(f"alignas(8) const unsigned char {model_name}[] = {{\n")

            # Write the model data as hexadecimal values
            for i, byte in enumerate(model_content):
                f.write(f"0x{byte:02x}, ")
                if (i + 1) % 12 == 0:  # 12 bytes per line
                    f.write("\n")

            f.write('\n};\n')
            f.write(f"const int {model_name}_len = {len(model_content)};\n")

    def build_autoencoder(self):
        # Define the encoder
        inputs = tf.keras.layers.Input(
            shape=(self.sequence_length, self.feature_size))

        # Add batch normalization at the input
        x = inputs
        # Encoder
        for i, units in enumerate(self.encoder_units):
            x = tf.keras.layers.LSTM(
                units,
                return_sequences=(i != len(self.encoder_units) - 1),
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout=0.1,
                recurrent_dropout=0.1,
                name=f'encoder_lstm_{i}'
            )(x)

        # Bottleneck
        encoded = x

        # Decoder
        x = tf.keras.layers.RepeatVector(self.sequence_length)(encoded)

        for i, units in enumerate(self.decoder_units):
            x = tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout=0.1,
                recurrent_dropout=0.1,
                name=f'decoder_lstm_{i}'
            )(x)

        # Output layer with proper scaling
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.feature_size, activation='tanh'), name="Dense_Output"
        )(x)

        model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,  # Fixed initial learning rate
            clipnorm=1.0  # Gradient clipping
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        return model

    def export_waterfall_lstm(input_model, input_size, input_layers_name, output_layers_name, units, enable_dense=False, features=3):
        # Flattened input size: input step + hidden states + cell states for each LSTM layer
        flattened_input_size = input_size + sum(units) * 2

        # Define the flattened input
        flattened_input = tf.keras.Input(
            shape=(flattened_input_size,), name='Flattened_Input')

        # Split the input using tf.slice and Lambda layers
        # Step Input (current input to the LSTM network)
        step_input = tf.keras.layers.Lambda(
            lambda x: tf.slice(x, [0, 0], [-1, input_size]),
            name='Slice_Input'
        )(flattened_input)

        # Hidden and cell states for each LSTM layer
        hidden_states_inputs = []
        cell_states_inputs = []
        offset = input_size
        for i, unit in enumerate(units):
            hidden_state = tf.keras.layers.Lambda(
                lambda x, offset=offset, unit=unit: tf.slice(
                    x, [0, offset], [-1, unit]),
                name=f'Slice_Hidden_State_{i}'
            )(flattened_input)
            cell_state = tf.keras.layers.Lambda(
                lambda x, offset=offset +
                unit, unit=unit: tf.slice(x, [0, offset], [-1, unit]),
                name=f'Slice_Cell_State_{i}'
            )(flattened_input)
            hidden_states_inputs.append(hidden_state)
            cell_states_inputs.append(cell_state)
            offset += unit * 2

        # Define LSTM cells and process inputs
        lstm_cells = []
        new_hidden_states = []
        new_cell_states = []
        lstm_output = step_input

        for i, unit in enumerate(units):
            lstm_cell = tf.keras.layers.LSTMCell(
                unit, name=f'{output_layers_name}_Cell_{i}')
            lstm_cells.append(lstm_cell)

            # Process current layer
            lstm_output, [new_hidden_state, new_cell_state] = lstm_cell(
                lstm_output, states=[
                    hidden_states_inputs[i], cell_states_inputs[i]]
            )

            # Collect the updated states
            new_hidden_states.append(new_hidden_state)
            new_cell_states.append(new_cell_state)

        if enable_dense:
            # Extract the final Dense layer from the original model
            dense_layer = None
            for layer in input_model.layers:
                if isinstance(layer, tf.keras.layers.TimeDistributed) and isinstance(layer.layer, tf.keras.layers.Dense):
                    dense_layer = layer.layer
                    break
            
            # Apply the Dense layer to the LSTM output
            final_output = dense_layer(lstm_output)
        else:
            final_output = lstm_output

        # Concatenate the final output and updated states into a single output
        flattened_output = tf.keras.layers.Concatenate(name='Flattened_Output')([
            final_output,
            *new_hidden_states,
            *new_cell_states
        ])

        # Create the inference model
        inference_model = tf.keras.Model(
            inputs=flattened_input, outputs=flattened_output)

        # Load weights from the training model
        for i, cell in enumerate(lstm_cells):
            training_layer = input_model.get_layer(
                name=f'{input_layers_name}_{i}')
            cell.set_weights(training_layer.get_weights())

        if enable_dense:
            # Load weights for the Dense layer if enabled
            training_dense_layer = input_model.get_layer(name='Dense_Output')
            dense_layer.set_weights(training_dense_layer.get_weights())

        return inference_model
    def export_encoder_decoder(self): 
        '''
        expected shape of the model
        input->encoder->RepeatVector->decoder->output
        where encoder and decoder are lstm layers defined by the encoder_units and decoder_units, but
        decoder also has a Dense layer at the end. 

        >>> trainer.model.layers
        [<InputLayer name=input_layer, built=True>, 
        <LSTM name=encoder_lstm_0, built=True>, 
        <LSTM name=encoder_lstm_1, built=True>, 
        <RepeatVector name=repeat_vector, built=True>, 
        <LSTM name=decoder_lstm_0, built=True>, 
        <LSTM name=decoder_lstm_1, built=True>, 
        <TimeDistributed name=time_distributed, built=True>]

        TimeDistributed is a wrapper for the final Dense layer
        '''

        encoder = LSTMModelTrainer.export_waterfall_lstm(
            self.model, self.feature_size, "encoder_lstm", "encoder_lstmCell", self.encoder_units)
        decoder = LSTMModelTrainer.export_waterfall_lstm(
            self.model, self.encoder_units[-1], "decoder_lstm", "decoder_lstmCell", self.decoder_units,True)


        return encoder, decoder

    def generate_timeseries(self, n_points=None,  randomness=0.15, noise_mean=0, noise_std_factor=0.01, noise_corr_factor=0.5, range_=range(3,6)):
        """
        Generate a multivariate timeseries with trend, seasonality, and noise components.

        Args:
            n_points (int): Number of data points to generate.
            feature_size (int): Number of features (dimensions) in the time series.
            randomness (float): Randomness factor for seasonality.
            noise_mean (float): Mean value for noise.
            noise_std_factor (float): Standard deviation factor for noise.
            noise_corr_factor (float): Correlation factor for noise.

        Returns:
            numpy.ndarray: Generated multivariate time series of shape (n_points, feature_size).
        """
        if n_points is None:
            n_points = self.sequence_length
        mu = 1
        sigma = randomness

        multivariate_series = []

        for _ in range(self.feature_size):
            # Flat trend (no slope)
            trend = FlatTrend(0)

            # Complex sinusoidal seasonality
            seasonality = SinusoidalSeasonality(
                amplitude=0.0, period=timedelta(seconds=1))
            for i in range_:
                seasonality += SinusoidalSeasonality(
                    amplitude=1 / (pow(i, 2)) * random.gauss(mu, sigma),
                    period=timedelta(seconds=random.gauss(
                        mu, sigma) * 3 * pow(i, 2))
                )

            # Red noise component
            noise = RedNoise(
                mean=noise_mean,
                std=noise_std_factor * random.gauss(mu, sigma),
                correlation=noise_corr_factor * random.gauss(mu, sigma)
            )

            # Combine trend, seasonality, and noise
            timeseries = trend + seasonality + noise

            # Generate dummy time points (for compatibility with `generate`)
            time_points = [datetime.now() + timedelta(seconds=i)
                           for i in range(n_points)]

            # Generate the values for this feature
            ts_values = timeseries.generate(time_points=time_points)

            multivariate_series.append(ts_values)

        # Stack all features to shape (n_points, feature_size)
        multivariate_series = np.stack(
            multivariate_series, axis=-1).reshape(1, n_points,   self.feature_size)

        return multivariate_series

    # not too useful fot the lstm, can safely ignore
    def pretrain_model(self):
        synthetic_data = []

        # Generate more diverse training samples
        for _ in range(1):
            dataset = self.generate_timeseries(
                n_points=self.sequence_length * 10,
                randomness=random.uniform(0.1, 0.3),
                noise_std_factor=0.00,  # Added some noise
                noise_corr_factor=random.uniform(0.3, 0.7), range_=(3,4)
            )
            chunked_dataset = LSTMModelTrainer.chunk_time_series(
                 dataset, self.sequence_length)
            synthetic_data.extend(chunked_dataset)

        # Convert to numpy array
        synthetic_data = np.array(synthetic_data)

        # Normalize the data
        mean = np.mean(synthetic_data)
        std = np.std(synthetic_data)
        synthetic_data = (synthetic_data - mean) / (std + 1e-10)

        self.__train(self.model, synthetic_data, epochs=150,
                     batch_size=32, validation_split=0.2)
        return synthetic_data

if __name__ == "__main__":

    trainer = LSTMModelTrainer(feature_size=1, sequence_length=40, encoder_units=[
                            20, 10], decoder_units=[10, 20])
    encoder = LSTMModelTrainer.export_waterfall_lstm(
        trainer.model, 1, "encoder_lstm", "encoder_lstmCell", [20, 10])
    encoder.save("encoder.keras")
    LSTMModelTrainer.export_model_to_tflite_file(encoder, "encoder.tflite")
    # code.interact(local=locals())
    trainer.model.save("autoencoder.keras")
    synthetic_data = trainer.pretrain_model()
    
    synthetic_data = []
    dataset = trainer.generate_timeseries(
        n_points=trainer.sequence_length * 10,
        randomness=random.uniform(0.1, 0.3),
        noise_std_factor=0.00,  # Added some noise
        noise_corr_factor=random.uniform(0.3, 0.7), range_=(3,3)
    )
    chunked_dataset = LSTMModelTrainer.chunk_time_series(
            dataset, trainer.sequence_length)
    synthetic_data.extend(chunked_dataset)

        # Convert to numpy array
    synthetic_data = np.array(synthetic_data)    
    synthetic_data2 = []
    dataset = trainer.generate_timeseries(
        n_points=trainer.sequence_length * 10,
        randomness=random.uniform(0.1, 0.3),
        noise_std_factor=0.00,  # Added some noise
        noise_corr_factor=random.uniform(0.3, 0.7), range_=range(3,6)
    )
    chunked_dataset = LSTMModelTrainer.chunk_time_series(
            dataset, trainer.sequence_length)
    synthetic_data2.extend(chunked_dataset)

        # Convert to numpy array
    synthetic_data2 = np.array(synthetic_data2)

    b = trainer.model.predict(synthetic_data)
    c= trainer.model.predict(synthetic_data2)
    # code.interact(local=locals())
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))  # 2 rows, 1 column

    # Plot 1: Training Input vs Predicted
    axs[0].plot(synthetic_data.reshape(-1), label="Input Time Series Train", linestyle="--")
    axs[0].plot(b.reshape(-1), label="Predicted Time Series Train", linestyle="-")
    axs[0].set_title("Training: Input vs Predicted Time Series")
    axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Values")
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Testing Input vs Predicted
    axs[1].plot(synthetic_data2.reshape(-1), label="Input Time Series Test", linestyle="--")
    axs[1].plot(c.reshape(-1), label="Predicted Time Series Test", linestyle="-")
    axs[1].set_title("Testing: Input vs Predicted Time Series")
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("Values")
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()

    # trainer.model.summary()

    # model = trainer.build_autoencoder()


    # trainer.model.summary()
    # trainer.train_autoencoder(epochs=1)
    # print(synthetic_data.shape)


    # a = (trainer.generate_timeseries())
    # code.interact(local=locals())

    # print(trainer.feature_size)
    # model = trainer.build_autoencoder()
    # model.save("autoencoder.keras")
