import os
import json
import importlib

import torch.utils.data as data_utils
import torch.optim

import eval
import dataset_utils.LJSpeech_Dataset as dataset_object
import model.build as model_builder
import model.Loss as loss

import poptorch


def dataloader_make(poptorch_opts,
                    dataset,
                    batch_size,
                    num_workers):
    dataloader = poptorch.DataLoader(options=poptorch_opts,
                                     dataset=dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers)
    return dataloader


def save_model(epoch,
               model_state,
               optimizer_state,
               loss,
               dir_checkpoint,
               checkpoint_filename):
    os.makedirs(config["checkpoint_dir"],
                exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': loss
        }, os.path.join(config["checkpoint_dir"],
                        "checkpoint.pt"))


def training_loop_ipu(poptorch_model,
                      model,
                      train_loader,
                      eval_loader,
                      optimizer,
                      config):
    poptorch_model.train()
    for i in range(config["epoch"]):
        for batch_idx, data in enumerate(train_loader):
            _, _, loss = poptorch_model(data["phone"],
                                        data["mel"].permute(0, 2, 1),
                                        data["phone_seq_len"],
                                        data["mel_seq_len"])

        if i % config["checkpoint_step"] == 0:
            save_model(epoch=i,
                       model_state=model.state_dict(),
                       optimizer_state=optimizer.state_dict(),
                       loss=loss,
                       dir_checkpoint=config["checkpoint_dir"],
                       checkpoint_filename="checkpoint.pt")

        if (i % config["eval_step"]) == 0:
            eval.evaluate(model, config)
        pass


def main_ipu(config):

    loss_criterion = loss.TransformerTTSLossIPU()
    model = model_builder.build_model_with_loss(config=config,
                                                loss_criterion=loss_criterion)
    opts = poptorch.Options()
    opts.deviceIterations(10)

    dataset = dataset_object.LJSPeech_Dataset(data_file=config["data_file"],
                                              root_dir=config["root_dir"],
                                              separator='#',
                                              mel_dir=config["spectogram_dir"])

    dataset_len = len(dataset)
    train_set, eval_set = torch.utils.data.random_split(dataset,
                                                        [int(dataset_len * 0.8),
                                                         int(dataset_len * 0.2)]
                                                        )

    train_loader = dataloader_make(poptorch_opts=opts,
                                   dataset=train_set,
                                   batch_size=config["batch_size"],
                                   num_workers=3)

    eval_loader = dataloader_make(poptorch_opts=opts,
                                  dataset=eval_set,
                                  batch_size=config["batch_size"],
                                  num_workers=3)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["learning_rate"])

    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)

    training_loop_ipu(poptorch_model=poptorch_model,
                      model=model,
                      train_loader=train_loader,
                      eval_loader=eval_loader,
                      optimizer=optimizer,
                      config=config)


if __name__ == "__main__":

    config = None

    with open("config/hparam.json", 'r') as f:
        config = json.load(f)

    poptorch_is_here = importlib.util.find_spec("poptorh")
    main_ipu(config)
