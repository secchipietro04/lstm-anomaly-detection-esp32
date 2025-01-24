from flask import Flask, request, Response
import numpy as np
from Model.ModelTrainer import LSTMModelTrainer

app = Flask(__name__)

# Session storage
sessions = {}
PORT=3253
HOST='192.168.43.132'

# how many times the controller will send data to the server
# 
N_FETCHES=16
class Session:
    FEATURE_SIZE = 3
    SEQUENCE_LENGTH = 40
    ENCODER_UNITS = [16, 8]
    DECODER_UNITS = [8, 16]

    def __init__(self, session_id):
        self.session_id = session_id
        self.data = None
        self.is_trained = False
        self.counter = 0
        self.trainer = LSTMModelTrainer(
            feature_size=self.FEATURE_SIZE, 
            sequence_length=self.SEQUENCE_LENGTH, 
            encoder_units=self.ENCODER_UNITS, 
            decoder_units=self.DECODER_UNITS
        )

    def add_raw_data(self, raw_data):
        data = []
        lines = raw_data.split('\n')
        for line in lines:
            if line.strip():
                try:
                    numbers = [int(num) for num in line.split(',')]
                    data.append(numbers)
                except ValueError:
                    raise ValueError("Invalid number format")
        
        data = np.array([data]) / 32768.0  # Normalize data when added
        new_chunks = LSTMModelTrainer.chunk_time_series(data, self.SEQUENCE_LENGTH)
        if self.data is None:
            self.data = new_chunks
        else:
            self.data = np.concatenate((self.data, new_chunks), axis=0)

        self.is_trained = False  # Reset training flag when new data is added

    def get_data(self):
        return self.data

    def clear_data(self):
        self.data = None
        self.is_trained = False

    def prepare_model(self):
        if self.data is not None and not self.is_trained:
            self.trainer.dataset = self.data

# Route to initialize a new session or send raw data
@app.route('/dataframe', methods=['POST'])
def handle_post():
    session_id = str(len(sessions) + 1)
    session = Session(session_id)
    sessions[session_id] = session
    print(f"Session {session_id} created")
    return Response(session_id, content_type='text/plain; charset=ascii'), 200

# Route to add more data to an existing session
@app.route('/dataframe/<string:id>', methods=['POST'])
def handle_post_dataframe_id(id):
    session = sessions.get(id)
    data = request.get_data(cache=True)

    if not session:
        session = Session(id)
        sessions[id] = session
    
    if session.counter > N_FETCHES:
        return "0", 200  # Session full

    raw_data = data.decode('utf-8')
    try:
        session.add_raw_data(raw_data)
        session.counter += 1
        print(f"Session {id} updated with data of len: {len(raw_data)}")
        return str(len(session.get_data()))
    except ValueError:
        return "Error: Invalid number format", 400

# Route to fetch encoder model
@app.route('/model/dataframe/<string:id>/encoder', methods=['GET'])
def get_encoder(id):
    session = sessions.get(id)
    if not session:
        return f"Error: Session {id} not found", 404

    session.prepare_model()
    if not session.is_trained:
        session.trainer.train_autoencoder(epochs=1)
        session.is_trained = True

    encoder, decoder = session.trainer.export_encoder_decoder()
    LSTMModelTrainer.export_model_to_tflite_file(encoder, f"encoder_{id}.tflite")
    encoder_tflite = LSTMModelTrainer.export_model_to_tflite(encoder)
    return Response(encoder_tflite, content_type='application/octet-stream')

# Route to fetch decoder model
@app.route('/model/dataframe/<string:id>/decoder', methods=['GET'])
def get_decoder(id):
    session = sessions.get(id)
    if not session:
        return f"Error: Session {id} not found", 404

    session.prepare_model()
    if not session.is_trained:
        session.trainer.train_autoencoder(epochs=1)
        session.is_trained = True

    encoder, decoder = session.trainer.export_encoder_decoder()
    LSTMModelTrainer.export_model_to_tflite_file(decoder, f"decoder_{id}.tflite")
    decoder_tflite = LSTMModelTrainer.export_model_to_tflite(decoder)
    return Response(decoder_tflite, content_type='application/octet-stream')

# Route to clear all sessions
@app.route('/clear_sessions', methods=['POST'])
def clear_sessions():
    sessions.clear()
    return "All sessions cleared", 200

if __name__ == '__main__':
    app.run(debug=True, port=PORT, host=HOST)
