
import torch


def batch_max_len(data):
    max_len = data
    return max_len


def collate_fn(batch):

    mel_max_len = batch_max_len(batch['mel'])
    phone_max_len = batch_max_len(batch['phone'])

    batch['mel'] = mel_batch
    batch['phone'] = phone_batch
    return batch
