
import torch


def batch_max_len(batch, key: str):
    max_len = torch.tensor(
            [data[key].shape[-1] for data in batch],
            dtype=torch.int32).max()
    return max_len


def collate_fn(batch):

    batch_size = len(batch)
    mel_max_len = batch_max_len(batch, 'mel')
    phone_max_len = batch_max_len(batch, 'phone')
    print(f"mel: {mel_max_len} | phone: {phone_max_len}")

    for idx in range(batch_size):
        batch[idx]['mel'] = torch.nn.functional.pad(
                batch[idx]['mel'],
                pad=[0, mel_max_len-batch[idx]['mel_seq_len']],
                value=0)

        batch[idx]['phone'] = torch.nn.functional.pad(
                batch[idx]['phone'],
                pad=[0, phone_max_len-batch[idx]['phone_seq_len']],
                value=0)

    collated_batch = dict()

    collated_batch['mel'] = torch.stack([data['mel'] for data in batch])
    collated_batch['phone'] = torch.stack([data['phone'] for data in batch])

    collated_batch['mel_seq_len'] = torch.tensor(
            [data['mel_seq_len'] for data in batch],
            dtype=torch.int32)

    collated_batch['phone_seq_len'] = torch.tensor(
            [data['phone_seq_len'] for data in batch],
            dtype=torch.int32)

    return collated_batch
