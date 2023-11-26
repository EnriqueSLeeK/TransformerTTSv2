
import torch


@torch.no_grad()
def evaluate(model, eval_set, config):
    total_loss = 0
    batch_i = -1
    model.eval()
    for batch_idx, data in enumerate(eval_set):
        data["mel"] = data["mel"].permute(0, 2, 1)

        # Put an start of sequence on the mel
        data["mel_input"] = torch.cat(
                [torch.zeros(data["phone"].shape[0], 1, config["n_mel"]),
                    data["mel"][:, :-1, :]],
                dim=1)

        _, _, model_loss = model(phoneme=data["phone"].cuda(),
                                 mel_input=data["mel_input"].cuda(),
                                 mel=data["mel"].cuda(),
                                 phone_seq_len=data["phone_seq_len"].cuda(),
                                 mel_seq_len=data["mel_seq_len"].cuda())
        total_loss += model_loss.item()
        batch_i = batch_idx

    print(f"Loss Mean in eval: {total_loss / (batch_i + 1)}")
    return (total_loss / (batch_i + 1))


@torch.no_grad()
def evaluate_ipu(model, eval_set, config):
    total_loss = 0
    batch_i = -1
    model.eval()
    for batch_idx, eval_data in enumerate(eval_set):
        output, loss = model(phoneme=eval_data["phone"],
                             mel=eval_data["mel"],
                             phone_seq_len=eval_data["phone_seq_len"],
                             mel_seq_len=eval_data["mel_seq_len"])
        total_loss += loss
        batch_i = batch_idx
    print(f"Loss Mean in eval: {total_loss / (batch_i + 1)}")
