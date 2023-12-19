
import torch
import json
import librosa
import os
from hifi_gan.inference_e2e import direct_inference
import matplotlib.pyplot as plt
import numpy as np

import model.build as builder


def extract_data(checkpoint_file):
    return torch.load(checkpoint_file)


def export_data(mel, stop_token, out='output_dir'):
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots()

    img = librosa.display.specshow(mel, x_axis='time',
                                   y_axis='mel', sr=22050,
                                   fmax=8000, ax=ax)

    ax.set(title='Mel-frequency spectrogram')

    fig.savefig(os.path.join(out, "rightImage.png"))
    np.save(os.path.join(out, "specto.npy"), mel)
    plt.close()

    p = torch.sigmoid(stop_token).cpu()
    plt.plot(p[0])
    plt.savefig(os.path.join(out, "leftImage.png"))
    plt.close()


def inference_train(model):
    model = builder.wrap_inference_mode(model).cuda()
    mel, stop = model("Hello world!")
    mel = mel[:, 1:, :]
    mel = mel.permute(0, 2, 1)[0].cpu().numpy()
    export_data(mel, stop_token=stop)


def inference_test(model):
    checkpoint_file = os.path.join(config['checkpoint_dir'],
                                   'checkpoint.pt')
    if (os.path.exists(checkpoint_file)):
        data = extract_data(checkpoint_file)
        model.model.load_state_dict(data['model_state_dict'])

    mel, stop = model("Hello world!")
    mel = mel[:, 1:, :]
    mel = mel.permute(0, 2, 1)[0].cpu().numpy()

    export_data(mel, stop_token=stop)
    direct_inference(mel)


config = None


def inference_text(model, text):
    checkpoint_file = os.path.join(config['checkpoint_dir'],
                                   'checkpoint.pt')
    if (os.path.exists(checkpoint_file)):
        data = extract_data(checkpoint_file)
        model.model.load_state_dict(data['model_state_dict'])

    mel, stop = model(text)
    mel = mel[:, 1:, :]
    mel = mel.permute(0, 2, 1)[0].cpu().numpy()

    export_data(mel, stop_token=stop, out='static/images')
    direct_inference(mel)


def inference_main(config):
    inference_model = builder.build_inference_model(config).cuda()

    checkpoint_file = os.path.join(config['checkpoint_dir'],
                                   'checkpoint.pt')
    if (os.path.exists(checkpoint_file)):
        data = extract_data(checkpoint_file)
        inference_model.model.load_state_dict(data['model_state_dict'])

    inference_test(inference_model)


def load_model():
    global config
    with open('config/hparam.json', 'r') as f:
        config = json.load(f)
    inference_model = builder.build_inference_model(config).cuda()

    checkpoint_file = os.path.join(config['checkpoint_dir'],
                                   'checkpoint.pt')
    if (os.path.exists(checkpoint_file)):
        data = extract_data(checkpoint_file)
        inference_model.model.load_state_dict(data['model_state_dict'])

    return inference_model


if __name__ == "__main__":
    config = None
    with open('config/hparam.json', 'r') as f:
        config = json.load(f)
    inference_main(config)
