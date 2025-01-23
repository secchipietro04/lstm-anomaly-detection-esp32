from flask import Flask, request, Response
app = Flask(__name__)
from Model.ModelTrainer import LSTMModelTrainer
import numpy as np
sessions = {}

# session store
class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.data = None 
        self.counter=16
        self.trainer=LSTMModelTrainer(feature_size=3, sequence_length=40, encoder_units=[16, 8], decoder_units=[8, 16])
    def add_raw_data(self, raw_data):
        # Process the raw data into a list of lists (integers)
        data=[]
        lines = raw_data.split('\n')
        for line in lines:
            if line.strip():  # Only process non-empty lines
                try:
                    numbers = [int(num) for num in line.split(',')]
                    data.append(numbers)
                except ValueError:
                    raise ValueError("Invalid number format")
        data=np.array([data])
        new_chunks = LSTMModelTrainer.chunk_time_series(data, 40)
        if self.data is None:
            self.data = new_chunks
        self.data = np.concatenate((self.data, new_chunks), axis=0)
        
    def get_data(self):
        # Return the processed data (list of lists of integers)
        return self.data
    
    def clear_data(self):
        # Clear the session data
        self.data.clear()


# Route to initialize a new session or send raw data
@app.route('/dataframe', methods=['POST', "GET"])
def handle_post():
    # Get raw data from the POST request body

    # Create a new session ID (for example, a simple counter or UUID can be used)
    session_id = str(len(sessions) + 1)
    # Create and store a new Session object
    session = Session(session_id)
    sessions[session_id] = session
    
    print(f"Session {session_id} created with data")  # Debug print
    
    # Return success with the session ID (as plain text)
    return Response(session_id, content_type='text/plain; charset=ascii'), 200


# Route to add more data to an existing session
@app.route('/dataframe/<string:id>', methods=['POST'])
def handle_post_dataframe_id(id):
    # Ensure the session exists
    session = sessions.get(id)
    data=request.get_data(cache=True)
    
    if not session:
        session = Session(id)
        sessions[id] = session
    if session.counter==0:
        return "0", 200 #session full
    # Get raw data from the request body
    raw_data = data.decode('utf-8')
    try:
        # print(raw_data)
        # Add new raw data to the session
        session.add_raw_data(raw_data)
        session.counter-=1
        print(f"Session {id} updated with data of len:", len(raw_data))  # Debug print

        return str(len(session.get_data()))  # Return the number of processed data items
    except ValueError:
        return "Error: Invalid number format", 400

def prepare_model(session):
    session.data = session.data / 32768.0  # Normalize the data
    session.trainer.dataset = session.data

# Route to fetch data from an existing session (model logic)
@app.route('/model/dataframe/<string:id>/encoder', methods=['GET'])
def get_encoder(id):
    # Ensure the session exists
    session = sessions.get(id)
    if not session:
        return f"Error: Session {id} not found", 404
    
    # Retrieve session data (processed data)
    prepare_model(session)
    session.trainer.train_autoencoder(epochs=1)
    e,d=session.trainer.export_encoder_decoder()
    LSTMModelTrainer.export_model_to_tflite_file(e,"encoder.tflite") 
    LSTMModelTrainer.export_model_to_tflite_file(d,"decoder.tflite") 
    encoder_tflite = LSTMModelTrainer.export_model_to_tflite(e)
    # Return the model data as raw bytes
    return Response(encoder_tflite, content_type='application/octet-stream')
# Route to fetch data from an existing session (model logic)

@app.route('/model/dataframe/<string:id>/decoder', methods=['GET'])
def get_decoder(id):
    # Ensure the session exists
    session = sessions.get(id)
    if not session:
        return f"Error: Session {id} not found", 404
    
    # Retrieve session data (processed data)

    prepare_model(session)
    session.trainer.train_autoencoder(epochs=50)
    e,d=session.trainer.export_encoder_decoder()
    LSTMModelTrainer.export_model_to_tflite_file(e,"encoder.tflite") 
    LSTMModelTrainer.export_model_to_tflite_file(d,"decoder.tflite") 
    decoder_tflite = LSTMModelTrainer.export_model_to_tflite(d)
    # Return the model data as raw bytes
    return Response(decoder_tflite, content_type='application/octet-stream')


# Route to clear a session (optional)
@app.route('/clear_sessions', methods=['POST'])
def clear_sessions():
    sessions.clear()
    return "All sessions cleared", 200


if __name__ == '__main__':
    app.run(debug=True, port=3253, host='192.168.43.132')
