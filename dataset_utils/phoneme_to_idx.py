
import numpy as np

phoneme = ['<pad>', '<sos>', '<eos>',
           'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2',
           'AH0', 'AH1', 'AH2',
           'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
           'B',
           'CH',
           'D', 'DH',
           'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
           'EY2',
           'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1',
           'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1',
           'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
           'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V',
           'W', 'Y', 'Z', 'ZH',
           '!', '?', ' ', '"', '\'', '.', ',', '..']
# sos = start of sentence
# eos = end of sentence


mapped_phoneme = [pair for pair in zip(phoneme, range(len(phoneme)))]
mapped_phoneme = dict(mapped_phoneme)


def get_pad_index():
    return mapped_phoneme['<pad>']


def phoneme_to_idx(phoneme_sequence):
    idx_phoneme = []
    for phone in phoneme_sequence:
        idx_phoneme.append(mapped_phoneme[phone])
    return len(idx_phoneme), np.array(idx_phoneme)


def idx_to_phoneme(sequence):
    phoneme_list = []

    for idx in sequence:
        phoneme_list.append(phoneme[idx])
    return phoneme_list


if __name__ == "__main__":
    print(len(mapped_phoneme))
