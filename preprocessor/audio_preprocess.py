
import numpy as np
import librosa
import os


# Use librosa
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


def normalize(data: np.array, min: float, max: float):
    norm = (data - data.min()) / (data.max() - data.min())
    return norm * (max - min) + min


def export_mel(file_name: str,
               output_dir: str,
               mel_spectogram: np.ndarray):
    file = os.path.join(output_dir,
                        file_name.split('/')[-1].replace(".wav",
                                                         ".npy"))
    with open(file, "w"):
        np.save(file, mel_spectogram)


# Resample and export audio files
def preprocess_audio(audio_list: list,
                     sr: int,
                     frame_size: int,
                     hop_size: int,
                     n_mels: int,
                     max_duration: int,
                     output_dir="preprocessed_data"):

    os.makedirs(os.path.join(output_dir, "mel_spectrogram"), exist_ok=True)
    duration_file = "preprocessed_data/mel_original_duration.csv"

    # Empty the duration file if already exist
    if os.path.isfile(duration_file):
        open(duration_file, "w").close()

    with open(duration_file, "a") as f:
        for audio_file in audio_list:
            # Load and resample the audio
            waveform, sample_rate = librosa.load(audio_file, sr=sr)

            # Trim audio removing silence at the front and at the
            # end of the audio
            waveform, _ = librosa.effects.trim(waveform)

            mel = extract_log_mel_spectogram(waveform,
                                             frame_size,
                                             hop_size,
                                             n_mels)

            mel = normalize(mel, -1, 1)

            # mel = pad_audio(mel, (waveform.shape[0] // hop_size) + 1)
            # Mel shape => [mel bands, mel frames]
            f.write(f"{audio_file.split('/')[-1]}|{mel.shape[1]}\n")

            export_mel(audio_file,
                       os.path.join(output_dir, "mel_spectrogram"),
                       mel)
