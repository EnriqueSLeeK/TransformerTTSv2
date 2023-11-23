import torch


# This will be used in stop token prediction
def stop_from_seq_lengths(
        sequence_lengths: torch.Tensor,
        max_length: int) -> torch.FloatTensor:
    # (batch_size, max_length)
    stop_token = (~(mask_from_seq_length(sequence_lengths,
                                         max_length))).float()
    return stop_token


# Generate mask
def mask_from_seq_length(
        sequence_lengths: torch.Tensor,
        max_length: int
        ) -> torch.BoolTensor:
    """
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


if __name__ == "__main__":
    mask = stop_from_seq_lengths(torch.Tensor([2, 2, 3]), 4)
    mask2 = stop_from_seq_lengths(torch.Tensor([4, 4, 4]), 4)
    # mask2 = stop_from_seq_lengths(torch.Tensor([2, 2, 3]), 4).float()
    # print(mask.view(-1, 1))
    # print(nnf.binary_cross_entropy(mask2, mask, reduction='mean'))
    # print(mask - mask2)
    print(mask.shape)
    print(mask)
    # print(type(mask))
