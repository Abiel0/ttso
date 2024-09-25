from flask import Flask, request, jsonify, send_file
from gradio_client import Client
from flask_cors import CORS
import base64
import os
import tempfile

app = Flask(__name__)
CORS(app)
# Initialize the client
client = Client("OpenSound/EzAudio")

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/generate_audio', methods=['POST'])
def generate_audio():

    data = request.json
    text = data.get('text', '')

    # Generate audio
    result = client.predict(
        text,  # text
        8,  # length
        5,  # guidance_scale
        0.75,  # guidance_rescale
        50,  # ddim_steps
        1,  # eta
        0,  # random_seed
        True,  # randomize_seed
        api_name="/generate_audio"
    )

    # Encode the audio file to base64
    with open(result, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

    # Remove the temporary file
    os.remove(result)

    return jsonify({"audio": encoded_audio})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)