import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import StockDataset
from model import PredictModel

PREPRO_DIR = './prepro'
RESULT_DIR = './result'


def get_ext_parser():
    parser = ArgumentParser()

    # train, valid, test split ratio which unit is "Y" from start date specified above.
    # i.e. "811" means that periods of train data, valid data and test data are 8Y, 1Y, and 1Y respectively.
    parser.add_argument('--split_ratio', type=str, default='811')

    # data period to be given to the model.
    parser.add_argument('--m', type=int, default=10)

    # data period to be used for prediction.
    parser.add_argument('--n', type=int, default=5)

    # CUDA Arguments
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--cuda_number', type=int, default=0)

    # Training Arguments
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--display_step', type=int, default=500)

    # Model Arguments
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=3)

    return parser


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, true):
        return torch.sqrt(self.mse_loss(pred, true))


if __name__ == '__main__':
    parser = get_ext_parser()
    args = parser.parse_args()

    split_ratio = args.split_ratio
    m = args.m
    n = args.n

    code = 'AAPL'

    split_ratio = list(map(int, list(split_ratio)))
    assert len(split_ratio) == 3
    train_period, valid_period, test_period = split_ratio

    prepro_path = os.path.join(PREPRO_DIR, f'split_{":".join(args.split_ratio)}')

    train_data_path = os.path.join(prepro_path, 'train', f'{code}.csv')
    valid_data_path = os.path.join(prepro_path, 'valid', f'{code}.csv')

    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)

    trainset = StockDataset(train_data, m, n)
    validset = StockDataset(valid_data, m, n)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(f"cuda:{args.cuda_number}" if torch.cuda.is_available() and args.use_gpu else "cpu")

    model = PredictModel(args.input_dim, args.hidden_dim, device)
    model = model.to(device)

    rmse_loss = RMSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_path = f'./result/{code}_m_{m}_n_{n}_split_{":".join(args.split_ratio)}.pt'
    loss_path = f'./result/{code}_m_{m}_n_{n}_split_{":".join(args.split_ratio)}_loss.png'

    best_valid_loss = np.Inf
    early_stop_count = 0

    training_losses = []
    valid_losses = []

    for epoch in tqdm(range(args.epoch), desc='Training'):
        print(f'============== TRAIN ON THE {epoch + 1}-th EPOCH ==============')
        training_loss = 0.0
        model.train()
        train_step = 0

        for step, (x, y, _) in enumerate(train_loader):
            train_step += 1
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out, mu, logvar = model(x, m, n)

            reconstruction_loss = rmse_loss(out[:, 1:], y)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = reconstruction_loss + kld_loss

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            if step % args.display_step == 0:
                print(f'[step {step}]' + '=' * (step // args.display_step))

        training_loss /= train_step
        training_losses.append(training_loss)
        print(f'[Epoch {epoch + 1}] Training Loss (RMSE) : {training_loss:.5f}')

        print('============== EVALUATION ON TEST DATA ==============')
        valid_loss = 0.0
        model.eval()
        valid_step = 0

        with torch.no_grad():
            for step, (x, y, _) in enumerate(train_loader):
                valid_step += 1
                x, y = x.to(device), y.to(device)

                out, mu, logvar = model(x, m, n)

                reconstruction_loss = rmse_loss(out[:, 1:], y)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = reconstruction_loss + kld_loss

                valid_loss += loss.item()

        valid_loss /= valid_step
        valid_losses.append(valid_loss)
        print(f'[Epoch {epoch + 1}] Valid Loss (RMSE) : {valid_loss:.5f}')

        if valid_loss < best_valid_loss:
            early_stop_count = 0
            print(f'[Epoch {epoch + 1}] {best_valid_loss:.5f} -> {valid_loss:.5f} Save Model...')
            best_valid_loss = valid_loss
            torch.save(model, model_path)
        else:
            early_stop_count += 1
            if early_stop_count == 50:
                break

    plt.plot(training_losses)
    plt.plot(valid_losses)
    plt.title('Model Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Valid'], loc='upper right')
    plt.savefig(loss_path)
