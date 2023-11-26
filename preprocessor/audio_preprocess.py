
import os
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import librosa

# Audio preprocessing taken from
# https://github.com/jik876/hifi-gan/blob/master/meldataset.py#L49-L72
# This is needed compability purpose

MAX_WAV_VALUE = 32768.0

mel_basis = {}
hann_window = {}


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


# Use librosa
def mel_spectrogram(y,
                    n_fft,
                    num_mels,
                    sampling_rate,
                    hop_size,
                    win_size,
                    fmin,
                    fmax,
                    center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def extract_log_mel_spectogram(waveform: np.ndarray,
                               n_fft: int,
                               hop_length: int,
                               n_mels: int):
    mel = librosa.feature.melspectrogram(y=waveform,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         n_mels=n_mels)

    mel = np.abs(mel)
    return librosa.power_to_db(mel)


def export_mel(file_name: str,
               output_dir: str,
               mel_spectogram: np.ndarray):
    file = os.path.join(output_dir,
                        file_name.split('/')[-1].replace(".wav",
                                                         ".npy"))
    with open(file, "w"):
        np.save(file, mel_spectogram)


# Resample and export audio files
def preprocess_audio(audio_list,
                     config):

    os.makedirs(os.path.join(config["spectogram_dir"],
                             "mel_spectrogram"),
                exist_ok=True)

    duration_file = "preprocessed_data/mel_original_duration.csv"

    # Empty the duration file if already exist
    if os.path.isfile(duration_file):
        open(duration_file, "w").close()

    with open(duration_file, "a") as f:
        for audio_file in audio_list:
            # Load and resample the audio
            waveform, sample_rate = librosa.load(audio_file,
                                                 sr=config['sample_rate'])
            waveform = waveform / MAX_WAV_VALUE
            waveform = normalize(waveform) * 0.95

            # Trim audio removing silence at the front and at the
            # end of the audio
            waveform, _ = librosa.effects.trim(waveform)

            waveform = torch.FloatTensor(waveform)
            waveform = waveform.unsqueeze(0)

            mel = mel_spectrogram(waveform,
                                  config['n_fft'],
                                  config['n_mel'],
                                  config['sample_rate'],
                                  config['hop_size'],
                                  config['win_size'],
                                  config['fmin'],
                                  config['fmax'],
                                  center=False)

            # mel = pad_audio(mel, (waveform.shape[0] // hop_size) + 1)
            # Mel shape => [mel bands, mel frames]
            f.write(f"{audio_file.split('/')[-1]}|{mel.shape[1]}\n")

            export_mel(audio_file,
                       os.path.join(config["spectogram_dir"],
                                    "mel_spectrogram"),
                       mel)
