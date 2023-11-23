import model.TransformerTTS as model_builder
import dataset_utils.phoneme_to_idx as pidx
import torch.nn as nn


# Build the encoder component
def build_encoder(d_model: int, n_head: int,
                  n_layer: int, dim_feedforward: int):
    encoder = nn.TransformerEncoderLayer(d_model=d_model,
                                         nhead=n_head,
                                         batch_first=True,
                                         dim_feedforward=dim_feedforward)

    return nn.TransformerEncoder(encoder_layer=encoder, num_layers=n_layer)


# Build the decoder component
def build_decoder(d_model: int, n_head: int,
                  n_layer: int, dim_feedforward: int):
    decoder = nn.TransformerDecoderLayer(d_model=d_model,
                                         nhead=n_head,
                                         batch_first=True,
                                         dim_feedforward=dim_feedforward)

    return nn.TransformerDecoder(decoder_layer=decoder, num_layers=n_layer)


# Build the TransformerTTS model
def build_model(config: dict):

    encoder = build_encoder(config["d_model"],
                            config["n_head"],
                            config["n_layer"],
                            config["dim_feedforward"])

    decoder = build_decoder(config["d_model"],
                            config["n_head"],
                            config["n_layer"],
                            config["dim_feedforward"])

    vocab_size = len(pidx.mapped_phoneme) + 1

    model = model_builder.TransformerTTS(encoder=encoder,
                                         decoder=decoder,
                                         vocab_size=vocab_size,
                                         config=config)
    return model


def build_model_with_loss(config: dict, loss_criterion):
    model_without_loss = build_model(config)

    model = model_builder.TransformerTTSWithLoss(model=model_without_loss,
                                                 loss_criterion=loss_criterion)
    return model


def build_inference_model(config: dict):

    model = build_model(config)

    model_inference = model_builder.TTSInference(model)
    return model_inference
