import torch.nn as nn


class MelLinear(nn.Module):
    def __init__(self, hidden_dim: int, mel_dim: int):
        super(MelLinear, self).__init__()
        self.linear = nn.Linear(hidden_dim, mel_dim)

    def forward(self, x):
        return self.linear(x)


class StopLinear(nn.Module):
    def __init__(self, hidden_dim: int):
        super(StopLinear, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear(x)


# A convolutional layer used in the Postnet class
class PostNetConv(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 dropout: float = 0.3,
                 activation=nn.Tanh()):

        super(PostNetConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim,
                              out_channels=output_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              dilation=1,
                              stride=1)

        self.batchNorm = nn.BatchNorm1d(output_dim)
        self.activation = activation

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


# PostNet of the neural network
class PostNet(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 mel_dim: int,
                 kernel_size: int,
                 dropout: float,
                 stride: int = 1,
                 dilation: int = 1):
        super(PostNet, self).__init__()

        self.conv_first = PostNetConv(input_dim=mel_dim,
                                      output_dim=hidden_dim,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      dropout=dropout)

        self.conv_list = nn.ModuleList([
            PostNetConv(input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        dropout=dropout)

            for _ in range(4)
            ])

        self.conv_final = PostNetConv(input_dim=hidden_dim,
                                      output_dim=mel_dim,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      dropout=dropout)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.conv_first(x)

        for conv in self.conv_list:
            x = conv(x)

        x = self.conv_final(x)
        x = x.permute(0, 2, 1)

        return x
