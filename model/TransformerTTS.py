import torch
import torch.nn as nn
import model.PostNet as postnet
import model.PreNet as prenet
import numpy as np
from g2p_en import G2p

from dataset_utils.phoneme_to_idx import phoneme_to_idx
from dataset_utils.mask_from_seq_len import stop_from_seq_lengths
from dataset_utils.mask_from_seq_len import mask_from_seq_length


class NetWorkPreNet(nn.Module):
    def __init__(self,
                 config: dict):
        super(NetWorkPreNet, self).__init__()

        self.prenet_encoder = prenet.EncoderPreNet(embed_dim=config["embed_dim"],
                                                   kernel_size=config["prenet_encoder_kernel"],
                                                   out_dim=config["prenet_out"],
                                                   dropout=config["dropout_prenet"])

        self.prenet_decoder = prenet.DecoderPreNet(input_dim=config["n_mel"],
                                                   hidden_dim=config["prenet_out"],
                                                   dropout=config["dropout_prenet"])

        self.max_mel_time = config["max_mel_frames"]

        self.positional_encoding = nn.Embedding(num_embeddings=self.max_mel_time,
                                                embedding_dim=config["pos_embed_dim"])

    def mask(self, mask_shape: tuple):
        mask = torch.full(mask_shape, float('-inf'))
        return torch.triu(mask, diagonal=1)

    def pad_mask(self,
                 mask_shape: tuple,
                 sequence_len: torch.Tensor,
                 max_len: int,
                 device):
        pad_mask = torch.zeros(
                mask_shape,
                device=device
                ).masked_fill(
                        ~mask_from_seq_length(sequence_len, max_len),
                        float('-inf')
                        ).float()
        return pad_mask

    def forward(self,
                phoneme,
                mel,
                phone_seq_len,
                mel_seq_len):

        # phoneme dim => [batch_size, seq_len, embed]
        # mel => [batch_size, mel_frames, mel_bands]

        encoder_input = self.prenet_encoder(phoneme)
        decoder_input = self.prenet_decoder(mel)

        # Using a simplified posticional encoding
        pos_code = self.positional_encoding(
                torch.arange(self.max_mel_time).to(mel.device))

        # Apply positional encoding to the inputs
        # We will use the embedding layer directly to do this
        encoder_input = encoder_input + pos_code[: encoder_input.shape[1]]
        decoder_input = decoder_input + pos_code[: mel.shape[1]]

        src_mask = self.mask(mask_shape=(phoneme.shape[1],
                                         phoneme.shape[1]))
        pad_src_mask = self.pad_mask(mask_shape=(phoneme.shape[0],
                                                 phoneme.shape[1]),
                                     sequence_len=phone_seq_len,
                                     max_len=phoneme.shape[1],
                                     device=phoneme.device)

        tgt_mask = self.mask(mask_shape=(mel.shape[1],
                                         mel.shape[1]))
        pad_tgt_mask = self.pad_mask(mask_shape=(mel.shape[0],
                                                 mel.shape[1]),
                                     sequence_len=mel_seq_len,
                                     max_len=mel.shape[1],
                                     device=mel.device)

        mem_mask = self.mask(mask_shape=(mel.shape[1],
                                         phoneme.shape[1]))

        return {"encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "src_mask": src_mask.cuda(),
                "memory_mask": mem_mask.cuda(),
                "target_mask": tgt_mask.cuda(),
                "pad_src_mask": pad_src_mask.cuda(),
                "pad_tgt_mask": pad_tgt_mask.cuda()
                }


class NetWorkPostNet(nn.Module):
    def __init__(self,
                 config: dict):
        super(NetWorkPostNet, self).__init__()

        self.stop_linear = postnet.StopLinear(hidden_dim=config["linear_dim"])

        self.mel_linear = postnet.MelLinear(hidden_dim=config["linear_dim"],
                                            mel_dim=config["n_mel"])

        self.postnet = postnet.PostNet(hidden_dim=config["post_dim"],
                                       mel_dim=config["n_mel"],
                                       kernel_size=config["kernel_post"],
                                       dropout=config["dropout_post"])

        self.dropout = nn.Dropout(config["dropout_post"])

    def forward(self, x):

        mel_projection = self.mel_linear(x)
        mel_construction = self.postnet(mel_projection)

        stop = self.stop_linear(x)

        return (mel_projection,
                mel_projection + mel_construction,
                stop)


class TransformerTTS(nn.Module):
    def __init__(self,
                 encoder: nn.TransformerEncoder,
                 decoder: nn.TransformerDecoder,
                 vocab_size: int,
                 config: dict):
        super(TransformerTTS, self).__init__()

        self.embedding = nn.Embedding(vocab_size,
                                      config["embed_dim"])

        self.network_prenet = NetWorkPreNet(config)

        self.network_postnet = NetWorkPostNet(config)

        self.mem_norm = nn.LayerNorm(config["linear_dim"])

        self.encoder = encoder
        self.decoder = decoder

    def encode(self,
               phoneme_embed,
               attn_mask,
               key_padding_mask):
        x = self.encoder(src=phoneme_embed,
                         mask=attn_mask,
                         src_key_padding_mask=key_padding_mask)
        return x

    def decode(self,
               mel,
               encoded,
               mem_mask=None,
               tgt_mask=None,
               pad_mem_mask=None,
               pad_tgt_mask=None):
        x = self.decoder(tgt=mel,
                         memory=encoded,
                         tgt_mask=tgt_mask,
                         memory_mask=mem_mask,
                         tgt_key_padding_mask=pad_tgt_mask,
                         memory_key_padding_mask=pad_mem_mask)
        return x

    def forward(self,
                phoneme,
                mel,
                phone_seq_len,
                mel_seq_len):

        embed_phoneme = self.embedding(phoneme)

        data = self.network_prenet(phoneme=embed_phoneme,
                                   mel=mel,
                                   phone_seq_len=phone_seq_len,
                                   mel_seq_len=mel_seq_len)

        encoded = self.encode(phoneme_embed=data["encoder_input"],
                              attn_mask=data["src_mask"],
                              key_padding_mask=data["pad_src_mask"])

        encoded = self.mem_norm(encoded)

        decoded = self.decode(mel=data["decoder_input"],
                              encoded=encoded,
                              mem_mask=data["memory_mask"],
                              tgt_mask=data["target_mask"],
                              pad_mem_mask=data["pad_src_mask"],
                              pad_tgt_mask=data["pad_tgt_mask"])

        mel_linear, mel_output, stop_token = self.network_postnet(decoded)

        bool_mel_mask = data["pad_tgt_mask"].ne(0).unsqueeze(-1).repeat(
                1, 1, mel.shape[2])

        mel_linear = mel_linear.masked_fill(bool_mel_mask, 0)
        mel_output = mel_output.masked_fill(bool_mel_mask, 0)

        stop_token = stop_token.masked_fill(
                bool_mel_mask[:, :, 0].unsqueeze(-1), 1e3).squeeze(2)

        return (mel_linear, mel_output, stop_token)


class TransformerTTSWithLoss(nn.Module):
    def __init__(self, model, loss_criterion):
        super(TransformerTTSWithLoss, self).__init__()
        self.model = model
        self.loss = loss_criterion

    def forward(self,
                phoneme,
                mel_input,
                mel,
                phone_seq_len,
                mel_seq_len):
        mel_linear, mel_out, token = self.model(phoneme=phoneme,
                                                mel=mel_input,
                                                phone_seq_len=phone_seq_len,
                                                mel_seq_len=mel_seq_len)

        # should i preprocess this and save it in a file?
        # Padded token generation
        stop_token = stop_from_seq_lengths(mel_seq_len,
                                           mel.shape[1])

        loss = self.loss(mel_linear=mel_linear,
                         predicted_mel=mel_out,
                         ground_truth_mel=mel,
                         stop_token_pred=token,
                         stop_token_expected=stop_token)

        return (mel_linear, mel_out, loss)


class TTSInference(nn.Module):
    def __init__(self, model):
        super(TTSInference, self).__init__()
        self.model = model

    def preprocess_text(self,
                        text: str,
                        device: str):
        g2p = G2p()
        phoneme = g2p(text)
        g2p(text).append('<eos>')
        seq_len, phoneme_idx = phoneme_to_idx(phoneme)

        # Converting to tensor and moving to the right device
        seq_len = torch.tensor(seq_len).unsqueeze(0).to(device)
        phoneme_idx = torch.from_numpy(np.array([phoneme_idx])).to(device)

        return (seq_len, phoneme_idx)

    def mel_add_start_token(self, n_mel: int, device: str):
        mel = torch.zeros((1, 1, n_mel), device=device)
        mel_len = torch.tensor(1).unsqueeze(0).to(device)
        return mel_len, mel

    @torch.no_grad()
    def forward(self,
                text: str,
                max_length: int = 800,
                stop_token_threshold: int = 1e3,
                n_mel: int = 80,
                device: str = 'cuda'):
        self.model.eval()
        # Prepare text
        seq_len, phoneme = self.preprocess_text(text, device)

        # Prepare mel
        mel_len, mel = self.mel_add_start_token(n_mel,
                                                device)

        stop_token = None
        # Inference of the audio spectogram
        for _ in range(max_length):
            mel_postnet, mel_linear, stop_token = self.model(phoneme,
                                                             mel,
                                                             seq_len,
                                                             mel_len)

            mel = torch.cat([mel, mel_postnet[:, -1:, :]], dim=1)

            # Check the stop_token and compare it to the stop_token_threshold
            if torch.sigmoid(stop_token[:, -1:]) > stop_token_threshold:
                break
            else:
                mel_len = torch.tensor(mel.shape[1]).unsqueeze(0).cuda()

        return mel, stop_token


if __name__ == "__main__":
    # print(NetWorkPreNet.pad_mask(None,
    #                              (4, 4),
    #                              torch.Tensor([0, 1, 2, 3]), 4))

    print(NetWorkPreNet.mask(None, (4, 4)))
    pass
