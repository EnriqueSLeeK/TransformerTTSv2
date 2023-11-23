
import os.path
import numpy as np
import torch.utils.data
import torch
#from dataset_utils.phoneme_to_idx import phoneme_to_idx
from phoneme_to_idx import phoneme_to_idx


# Both padded
# Mel specto dims => (Batch size, mel bands, mel frames)
# Phoneme => (batch size, max seq len)
class LJSPeech_Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_file: str,
                 root_dir: str,
                 separator: str = ' ',
                 mel_dir: str = None,
                 orig_dur: str = None):
        self.audio_files = []
        self.audio_orig_duration = dict()
        self.phoneme = []

        if orig_dur is not None:
            with open(os.path.join(root_dir, orig_dur)) as f:
                for line in f:
                    splitted = line.split('|')
                    self.audio_orig_duration[os.path.join(root_dir, mel_dir, splitted[0] + ".npy")] = int(splitted[1])

        with open(os.path.join(root_dir, data_file), 'r') as f:
            for line in f:
                splitted = line.split('|')

                if mel_dir:
                    self.audio_files.append(
                        os.path.join(root_dir, mel_dir, splitted[0] + ".npy"))

                splitted[1] = splitted[1].strip() + separator + "<eos>"
                self.phoneme.append(splitted[1].split(separator))

        self.default_duration = 0
        self.root_dir = root_dir

    def __len__(self):
        return len(self.phoneme)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]

        mel_spectogram = torch.from_numpy(np.load(audio_file))

        seq_len, idx = phoneme_to_idx(self.phoneme[index])
        phoneme = torch.from_numpy(idx)

        if self.audio_orig_duration:
            mel_len = self.audio_orig_duration[audio_file]
        else:
            mel_len = mel_spectogram.shape[1]

        if not self.default_duration:
            self.default_duration = mel_len

        return {"mel": mel_spectogram,
                "phone": phoneme,
                "phone_seq_len": seq_len,
                "mel_seq_len": mel_len}


if __name__ == "__main__":
    dataset = LJSPeech_Dataset(data_file="preprocessed_text.txt",
                               root_dir="preprocessed_data",
                               separator="#",
                               mel_dir="mel_spectrogram",
                               orig_dur="mel_original_duration.csv")

    train_set, test_set = torch.utils.data.random_split(dataset, [10480, 2620])

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=5,
                                               shuffle=False)

    for spectogram, pack in enumerate(train_loader):
        # print(pack["mel"])
        print(type(pack["mel"].permute(0, 2, 1)))
        # print(pack["phone"].shape)
        print("----------------------")
        break
        #time.sleep(1)
