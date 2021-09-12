import os
import pickle
import numpy as np
import platform
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('layer_640', nn.Linear(768, 640))
        self.encoder.add_module('layer_640_activation', nn.PReLU())
        self.encoder.add_module('layer_512', nn.Linear(640, 512))
        self.encoder.add_module('layer_512_activation', nn.PReLU())
        self.encoder.add_module('layer_384', nn.Linear(512, 384))
        self.encoder.add_module('layer_384_activation', nn.PReLU())
        self.encoder.add_module('layer_256', nn.Linear(384, 256))
        self.encoder.add_module('layer_256_activation', nn.PReLU())
        self.encoder.add_module('layer_128', nn.Linear(256, 128))
        self.encoder.add_module('layer_128_activation', nn.PReLU())
        self.encoder.add_module('layer_64', nn.Linear(128, 64))
        self.encoder.add_module('layer_64_activation', nn.Tanh())

        # Decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module('layer_128', nn.Linear(64, 128))
        self.decoder.add_module('layer_128_activation', nn.PReLU())
        self.decoder.add_module('layer_256', nn.Linear(128, 256))
        self.decoder.add_module('layer_256_activation', nn.PReLU())
        self.decoder.add_module('layer_384', nn.Linear(256, 384))
        self.decoder.add_module('layer_384_activation', nn.PReLU())
        self.decoder.add_module('layer_512', nn.Linear(384, 512))
        self.decoder.add_module('layer_512_activation', nn.PReLU())
        self.decoder.add_module('layer_640', nn.Linear(512, 640))
        self.decoder.add_module('layer_640_activation', nn.PReLU())
        self.decoder.add_module('layer_768', nn.Linear(640, 768))
        self.decoder.add_module('layer_768_activation', nn.Tanh())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_data(
        path_data: str,
):
    with open(path_data, 'rb') as f:
        return pickle.load(f)


def create_nn(
        path_model: str,
):
    cuda_available = torch.cuda.is_available()

    data = get_data('/media/otana/Remote HDD/data/imsim_predictions/predictions.pickle')
    data_len = len(data)

    data_train = data[:int(data_len * .1 // -1)]
    data_val = data[int(data_len * .1 // -1):]

    loader_train = DataLoader(
        dataset=data_train,
        batch_size=256,
        shuffle=True,
    )
    loader_val = DataLoader(
        dataset=data_val,
        batch_size=256,
        shuffle=True,
    )

    epochs = 1000
    appropriate_val_rate = 5e-2
    criterion = nn.MSELoss()

    model = NeuralNetwork()
    model = model.cuda() if cuda_available else model.cpu()

    optimizer = torch.optim.Adam(
        params=model.parameters(),
    )

    min_valid_loss = np.inf

    losses_train = []
    losses_val = []
    model_saved = False
    current_epoch = 0

    while not model_saved or current_epoch < epochs:
        model.train()
        loss_train = .0
        for futures in loader_train:
            futures = futures.cuda() if cuda_available else futures.cpu()

            optimizer.zero_grad()
            predicted = model(futures)

            loss = criterion(predicted, futures)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train = loss_train / len(loader_train)
        losses_train.append(loss_train)

        model.eval()
        loss_val = .0
        for futures in loader_val:
            futures = futures.cuda() if cuda_available else futures.cpu()

            predicted = model(futures)

            loss = criterion(predicted, futures)
            loss_val += loss.item()
        loss_val = loss_val / len(loader_val)
        losses_val.append(loss_val)

        output_epoch = f''
        output_epoch += f'Epoch {current_epoch + 1} \t\t '
        output_epoch += f'Training Loss: {loss_train} \t\t '
        output_epoch += f'Validation Loss: {loss_val} \t\t '
        print(output_epoch)

        appropriate_val = loss_train * appropriate_val_rate
        is_val_appropriate = True if abs(loss_train - loss_val) < appropriate_val else False
        if min_valid_loss > loss_val and is_val_appropriate:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{loss_val:.6f}) \t Saving The Model')
            min_valid_loss = loss_val
            # Saving layer
            torch.save(model.encoder, path_model)
            model_saved = True

        current_epoch += 1

    # Defining the Plot Style
    # plt.style.use('fivethirtyeight')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    #
    # Plotting the last 100 values
    # plt.plot(losses_train)
    # plt.plot(losses_val)
    # plt.savefig('model.png')

    # print(model)


if __name__ == '__main__':
    path = os.path.abspath('../models/vit_encoder_new.pickle')
    create_nn(path)
