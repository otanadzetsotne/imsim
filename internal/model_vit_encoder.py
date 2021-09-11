# TODO

import os
import pickle
import numpy as np
import platform
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(
            self,
            layer_input: int,
            layer_hidden: int,
    ):
        super(NeuralNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(layer_input, layer_hidden),
        )

        self.decoder = nn.Sequential(
            nn.Linear(layer_hidden, layer_input),
        )

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

    data = get_data('C:\\Users\\otana\\Разработка\\py\\imsim_predictions\\predictions.pickle')
    data_train = data[:5000]
    data_val = data[5000:5500]

    # data_train = data[:-10000]
    # data_val = data[-10000:]

    loader_train = DataLoader(
        dataset=data_train,
        batch_size=64,
        shuffle=True,
    )
    loader_val = DataLoader(
        dataset=data_val,
        batch_size=64,
        shuffle=True,
    )

    epochs = 15
    appropriate_val_rate = 1e-2
    criterion = nn.MSELoss()

    inputs = [640, 512, 384, 256, 128]
    input_last = 768
    for input_hidden in inputs:
        model = NeuralNetwork(input_last, input_hidden)
        model = model.cuda() if cuda_available else model.cpu()

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=1e-3,
            weight_decay=1e-8,
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
                torch.save(model.encoder, f'{path_model}/layer_{input_hidden}.pickle')
                model_saved = True

            current_epoch += 1

        input_last = input_hidden

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
    path = os.path.abspath('../models/')
    create_nn(path)
