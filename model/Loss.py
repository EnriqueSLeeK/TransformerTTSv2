import torch.nn as nn
import importlib

# Check if poptorch exist
poptorch_here = importlib.util.find_spec("poptorch")
if poptorch_here is not None:
    import poptorch


class TransformerTTSLossIPU(nn.Module):
    """
    Loss function from tacotron2
    Link: https://github.com/NVIDIA/tacotron2/blob/master/loss_function.py
    """

    def __init__(self,
                 criterion1=nn.MSELoss(),
                 criterion2=nn.BCEWithLogitsLoss()):
        super(TransformerTTSLossIPU, self).__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def ipu_loss(self,
                 mel_linear,
                 predicted_mel,
                 ground_truth_mel,
                 stop_token_pred,
                 stop_token_expected):

        mel_loss = self.criterion1(predicted_mel,
                                   ground_truth_mel) + \
                    self.criterion1(mel_linear,
                                    ground_truth_mel)

        stop_token_pred = stop_token_pred.view(-1, 1)
        stop_token_expected = stop_token_expected.view(-1, 1)

        stop_token_loss = self.criterion2(stop_token_pred,
                                          stop_token_expected)

        return poptorch.identity_loss(mel_loss + stop_token_loss,
                                      reduction="none")

    def forward(self,
                mel_linear,
                predicted_mel,
                ground_truth_mel,
                stop_token_pred,
                stop_token_expected):

        loss = self.ipu_loss(mel_linear=mel_linear,
                             predicted_mel=predicted_mel,
                             ground_truth_mel=ground_truth_mel,
                             stop_token_pred=stop_token_pred,
                             stop_token_expected=stop_token_expected)

        return loss


class TransformerTTSLossGpu(nn.Module):
    """
    Loss function from tacotron2
    Link: https://github.com/NVIDIA/tacotron2/blob/master/loss_function.py
    """

    def __init__(self,
                 criterion1=nn.MSELoss(),
                 criterion2=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(7.5))):
        super(TransformerTTSLossGpu, self).__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def gpu_loss(self,
                 mel_linear,
                 predicted_mel,
                 ground_truth_mel,
                 stop_token_pred,
                 stop_token_expected):

        mel_loss = self.criterion1(predicted_mel,
                                   ground_truth_mel) + \
                    self.criterion1(mel_linear,
                                    ground_truth_mel)

        stop_token_pred = stop_token_pred.view(-1, 1)
        stop_token_expected = stop_token_expected.view(-1, 1)

        stop_token_loss = self.criterion2(stop_token_pred,
                                          stop_token_expected)
        return mel_loss + stop_token_loss

    def forward(self,
                mel_linear,
                predicted_mel,
                ground_truth_mel,
                stop_token_pred,
                stop_token_expected):

        return self.gpu_loss(mel_linear=mel_linear,
                             predicted_mel=predicted_mel,
                             ground_truth_mel=ground_truth_mel,
                             stop_token_pred=stop_token_pred,
                             stop_token_expected=stop_token_expected)
