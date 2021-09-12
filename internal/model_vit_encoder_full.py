import pickle

import fire
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class ViTAutoEncoder(nn.Module):
    def __init__(self):
        super(ViTAutoEncoder, self).__init__()

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
        path_data: str,
        path_model: str,
        epochs: int = 100,
        epochs_mas: int = 1000,
        appropriate_val_rate: float = 1e-2,
):
    # Get features matrix
    data = get_data(path_data)
    data_split = int(len(data) * .1 // -1)

    # Split features
    data_train = data[:data_split]
    data_val = data[data_split:]

    loader_train = DataLoader(
        dataset=data_train,
        batch_size=2048,
        shuffle=True,
    )
    loader_val = DataLoader(
        dataset=data_val,
        batch_size=2048,
        shuffle=True,
    )

    # Loss function
    criterion = nn.MSELoss()

    # Initialize model
    model = ViTAutoEncoder()
    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    # Optimizer with default params
    optimizer = torch.optim.Adam(
        params=model.parameters(),
    )

    # Training loop params
    min_valid_loss = np.inf
    losses_train = []
    losses_val = []
    model_saved = False
    current_epoch = 0
    while not model_saved or current_epoch < epochs:
        # Turn on training
        model.train()
        loss_train = .0
        # Loop data batches
        for futures in loader_train:
            futures = futures.cuda() if torch.cuda.is_available() else futures.cpu()

            # Optimizer step
            optimizer.zero_grad()
            # Predict futures
            predicted = model(futures)
            # Loss
            loss = criterion(predicted, futures)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # Mean loss for each batch
        loss_train = loss_train / len(loader_train)
        losses_train.append(loss_train)

        # Turn on evaluation
        model.eval()
        loss_val = .0
        # Validate model
        for futures in loader_val:
            futures = futures.cuda() if torch.cuda.is_available() else futures.cpu()

            predicted = model(futures)

            loss = criterion(predicted, futures)
            loss_val += loss.item()
        loss_val = loss_val / len(loader_val)
        losses_val.append(loss_val)

        # Info
        output_epoch = f''
        output_epoch += f'Epoch {current_epoch + 1} \t\t '
        output_epoch += f'Training Loss: {loss_train} \t\t '
        output_epoch += f'Validation Loss: {loss_val} \t\t '
        print(output_epoch)

        # Appropriate train/validation loss value difference
        appropriate_val = loss_train * appropriate_val_rate
        # Is model appropriate
        is_val_appropriate = True if abs(loss_train - loss_val) < appropriate_val else False
        if min_valid_loss > loss_val and is_val_appropriate:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{loss_val:.6f}) \t Saving The Model')
            min_valid_loss = loss_val
            # Save encoder model
            torch.save(model.encoder, path_model)
            model_saved = True

        current_epoch += 1
        if current_epoch >= epochs_mas:
            break


if __name__ == '__main__':
    fire.Fire()
