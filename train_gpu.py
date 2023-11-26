import os
import json
import sys
import arg_parser.arg_parser as arg_parse

import torch.utils.data as data_utils
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

import eval
import dataset_utils.LJSpeech_Dataset as dataset_object
import dataset_utils.batch_collate_fn as collate
import model.build as model_builder
import model.Loss as loss


def extract_data(checkpoint_file):
    return torch.load(checkpoint_file)


def dataloader_make(dataset, batch_size):
    dataloader = data_utils.DataLoader(dataset,
                                       batch_size=batch_size,
                                       num_workers=2,
                                       shuffle=True,
                                       collate_fn=collate.collate_fn)
    return dataloader


def save_model(step,
               model_state,
               optimizer_state,
               train_loss,
               test_loss,
               dir_checkpoint,
               checkpoint_filename):
    os.makedirs(dir_checkpoint,
                exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'train_loss': train_loss,
        'test_loss': test_loss
        }, os.path.join(config["checkpoint_dir"],
                        "checkpoint.pt"))


def profile_gpu(model,
                train_loader,
                optimizer):
    iter_train = iter(train_loader)
    data = next(iter_train)
    with profile(activities=[ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True) as prof:

        _, _, model_loss = model(phoneme=data["phone"].cuda(),
                                 mel=data["mel"].permute(0, 2, 1).cuda(),
                                 phone_seq_len=data["phone_seq_len"].cuda(),
                                 mel_seq_len=data["mel_seq_len"].cuda())

        print(prof.key_averages().table(sort_by="self_cpu_memory_usage",
                                        row_limit=10))


def training_loop_gpu(model,
                      train_loader,
                      eval_loader,
                      optimizer,
                      config,
                      logger):

    checkpoint_file = os.path.join(config['checkpoint_dir'],
                                   'checkpoint.pt')
    i = 0
    k = 0
    loss_mean = 0.0

    if (os.path.exists(checkpoint_file)):
        data = extract_data(checkpoint_file)
        model.model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        i = int(data['step'])
        loss_mean = float(data['train_loss'])
        print("Model checkpoint load")

    epoch = int(i // 406.25)

    checkpoint_step = config["checkpoint_step"]
    model.train()
    print("Training")
    while (epoch <= config["epoch"]):

        for batch_idx, data in enumerate(train_loader):
            i += 1
            k += 1

            optimizer.zero_grad()

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

            model_loss.backward()
            optimizer.step()
            loss_mean += model_loss.item()

            if i % 500 == 0:
                print(f'{i} step_loss_mean: {loss_mean / k}')

                if i % checkpoint_step == 0:
                    print("Evaluation")
                    loss_mean = model_loss / checkpoint_step

                    eval_mean_loss = eval.evaluate(model, train_loader, config)
                    model.train()

                    save_model(step=i,
                               model_state=model.model.state_dict(),
                               optimizer_state=optimizer.state_dict(),
                               dir_checkpoint=config["checkpoint_dir"],
                               checkpoint_filename="checkpoint.pt",
                               train_loss=loss_mean,
                               test_loss=eval_mean_loss)

                    logger.add_scalar("Eval Loss",
                                      eval_mean_loss,
                                      global_step=i)
                    print("Checkpoint!")

                    loss_mean = 0.0
                    k = 0


def main_gpu(config):
    loss_criterion = loss.TransformerTTSLossGpu().cuda()

    model = model_builder.build_model_with_loss(config, loss_criterion).cuda()

    dataset = dataset_object.LJSPeech_Dataset(data_file=config["data_file"],
                                              root_dir=config["root_dir"],
                                              separator='#',
                                              mel_dir=config["spectogram_dir"],
                                              orig_dur=config["orig_duration"])

    dataset_len = len(dataset)
    train_set, eval_set = torch.utils.data.random_split(dataset,
                                                        [int(dataset_len * 0.8),
                                                         int(dataset_len * 0.2)]
                                                        )

    train_loader = dataloader_make(train_set,
                                   batch_size=config["batch_size"])

    eval_loader = dataloader_make(eval_set,
                                  batch_size=config["batch_size"])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["learning_rate"])

    os.makedirs(config['log_dir'], exist_ok=True)
    logger = SummaryWriter(config['log_dir'])

    training_loop_gpu(model=model,
                      train_loader=train_loader,
                      eval_loader=eval_loader,
                      optimizer=optimizer,
                      config=config,
                      logger=logger)


if __name__ == "__main__":

    config = None

    with open("config/hparam.json", 'r') as f:
        config = json.load(f)

    main_gpu(config)
