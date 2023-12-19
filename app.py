from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from inference_gpu import inference_text, load_model

app = Flask(__name__)

AUDIO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')

model = load_model()
os.makedirs(AUDIO_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')


@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    text = request.json['text']
    audio_filename = 'generated_audio.wav'
    inference_text(model, text)
    return jsonify({"audio_path": audio_filename})


@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
