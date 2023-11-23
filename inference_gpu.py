
import torch
import json
import os

import model.build as builder


def extract_data(checkpoint_file):
    return torch.load(checkpoint_file)


def inference(model):
    checkpoint_file = os.path.join(config['checkpoint_dir'],
                                   'checkpoint.pt')
    if (os.path.exists(checkpoint_file)):
        data = extract_data(checkpoint_file)
        model.model.load_state_dict(data['model_state_dict'])

    mel = model("Hello world!")
    print(mel)


def main(config):
    inference_model = builder.build_inference_model(config).cuda()

    inference(inference_model)


if __name__ == "__main__":
    config = None
    with open('config/hparam.json', 'r') as f:
        config = json.load(f)
    main(config)
