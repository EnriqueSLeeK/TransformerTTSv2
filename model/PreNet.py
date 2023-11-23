
import torch.nn as nn
import torch.nn.functional as nnf


class DecoderPreNet(nn.Module):
    # This section is responsible for projecting
    # the mel spectogram
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.5):
        super(DecoderPreNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = nnf.dropout(x,
                        p=self.dropout,
                        training=True)

        x = self.activation(self.fc2(x))
        x = nnf.dropout(x,
                        p=self.dropout,
                        training=True)

        return x


class ConvEncoderPreNet(nn.Module):
    # Conv part of the EncoderPrenet
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 dropout: float = 0.1):
        super(ConvEncoderPreNet, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size // 2,
                              dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channel)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class EncoderPreNet(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 kernel_size: int,
                 out_dim: int,
                 dropout: float):
        super(EncoderPreNet, self).__init__()

        self.convL1 = ConvEncoderPreNet(in_channel=embed_dim,
                                        out_channel=embed_dim,
                                        kernel_size=kernel_size)

        self.convL2 = ConvEncoderPreNet(in_channel=embed_dim,
                                        out_channel=embed_dim,
                                        kernel_size=kernel_size)

        self.convL3 = ConvEncoderPreNet(in_channel=embed_dim,
                                        out_channel=embed_dim,
                                        kernel_size=kernel_size)

        self.projection = nn.Linear(embed_dim,
                                    out_dim)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.convL1(x)
        x = self.convL2(x)
        x = self.convL3(x)

        x = x.permute(0, 2, 1)

        x = self.projection(x)

        return x
