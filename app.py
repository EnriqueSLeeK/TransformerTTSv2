from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from inference_gpu import inference_text, load_model

# Cria uma instância da aplicação Flask
app = Flask(__name__)

# Define o caminho para a pasta onde os arquivos de áudio serão salvos
AUDIO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')

model = load_model()
# Cria a pasta de áudio se ela não existir
os.makedirs(AUDIO_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')


@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    # Extrai o texto do pedido JSON e converte em áudio
    text = request.json['text']
    # TODO: Aqui será implementado o código para chamar o modelo TTS do repositório do GitHub e gerar o áudio
    # Supõe-se que o nome do arquivo de áudio gerado pelo TTS seja 'generated_audio.wav'
    audio_filename = 'generated_audio.wav'
    inference_text(model, text)
    # TODO: Aqui o arquivo de áudio gerado pelo TTS será salvo na pasta AUDIO_FOLDER
    # audio_path = os.path.join(AUDIO_FOLDER,
    #                          audio_filename)

    # Retorna o caminho relativo do arquivo de áudio
    return jsonify({"audio_path": audio_filename})


@app.route('/audio/<filename>')
def get_audio(filename):
    # Envia o arquivo de áudio solicitado do diretório AUDIO_FOLDER
    return send_from_directory(AUDIO_FOLDER, filename)


if __name__ == '__main__':
    # Inicia a aplicação com o modo de depuração ativado
    app.run(debug=True)
