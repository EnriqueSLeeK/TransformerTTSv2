
import torch
import json
import librosa
import sys
import os
from hifi_gan.inference_e2e import direct_inference
import matplotlib.pyplot as plt
import numpy as np

import model.build as builder


def extract_data(checkpoint_file):
    return torch.load(checkpoint_file)


def inference_test(model):
    out = 'output_dir'
    os.makedirs(out, exist_ok=True)

    checkpoint_file = os.path.join(config['checkpoint_dir'],
                                   'checkpoint.pt')
    if (os.path.exists(checkpoint_file)):
        data = extract_data(checkpoint_file)
        model.model.load_state_dict(data['model_state_dict'])

    mel = model("Hello world!")
    mel = mel.permute(0, 2, 1)[0].cpu().numpy()

    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel, x_axis='time',
                                   y_axis='mel', sr=22050,
                                   fmax=8000, ax=ax)

    ax.set(title='Mel-frequency spectrogram')

    fig.savefig(os.path.join(out, "mel_fig2.png"))
    np.save(os.path.join(out, "specto.npy"), mel)
    direct_inference(mel)

config = None

def inference_text(model, text):
    out = 'output_dir'
    os.makedirs(out, exist_ok=True)

    checkpoint_file = os.path.join(config['checkpoint_dir'],
                                   'checkpoint.pt')
    if (os.path.exists(checkpoint_file)):
        data = extract_data(checkpoint_file)
        model.model.load_state_dict(data['model_state_dict'])

    mel = model(text)
    mel = mel.permute(0, 2, 1)[0].cpu().numpy()

    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel, x_axis='time',
                                   y_axis='mel', sr=22050,
                                   fmax=8000, ax=ax)

    fig.colorbar(img, ax=ax, format='%+.3f dB')

    ax.set(title='Mel-frequency spectrogram')

    fig.savefig(os.path.join(out, "mel_fig2.png"))
    np.save(os.path.join(out, "specto.npy"), mel)
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
