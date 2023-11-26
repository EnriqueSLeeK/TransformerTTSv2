
import os
import json
from pathlib import Path
import arg_parser.arg_parser as Arg_parser

import preprocessor.text_preprocess as text_prep
import preprocessor.audio_preprocess as audio_prep


def get_file_names(dir: str, expr: str):
    return [str(file) for file in Path(dir).rglob(expr)]


def get_json_config(dir: str, config_file: str):
    data = None
    with open(os.path.join(dir, config_file), "r") as json_config:
        data = json.load(json_config)
    config = dict(data)
    return config


def main(arg_dict, config):

    os.makedirs(arg_dict["-output_dir"], exist_ok=True)

    wav_list = get_file_names(arg_dict["-dir"], "*.wav")
    text_transcript = os.path.join(arg_dict["-dir"],
                                   arg_dict["-transcript_file"])

    text_prep.text_and_preprocess(text_transcript,
                                  "|",
                                  output_dir=arg_dict["-output_dir"])

    audio_prep.preprocess_audio(wav_list, config)


if __name__ == "__main__":
    parser = Arg_parser.ArgParser()

    # dir stands for data directory
    # sr sample rate
    # config file for the hyperparams and settings
    parser.add_flag_argument("-dir", "LJSpeech-1.1")
    parser.add_flag_argument("-transcript_file", "metadata.csv")
    parser.add_flag_argument("-output_dir", "preprocessed_data")
    parser.add_flag_argument("-config_file", "hparam.json")
    parser.parse()

    arg_dict = parser.argument
    config = get_json_config("config", arg_dict["-config_file"])
    main(arg_dict, config)
