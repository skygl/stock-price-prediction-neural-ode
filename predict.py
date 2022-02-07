import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PREPRO_DIR = './prepro'
RESULT_DIR = './result'


def get_ext_parser():
    parser = ArgumentParser()

    # train, valid, test split ratio which unit is "Y" from start date specified above.
    # i.e. "811" means that periods of train data, valid data and test data are 8Y, 1Y, and 1Y respectively.
    parser.add_argument('--split_ratio', type=str, default='811')

    # "during training" data period to be given to the model.
    # it will be used for loading trained model
    parser.add_argument('--m', type=int, default=10)

    # "during training" data period to be used for prediction.
    # it will be used for loading trained model
    parser.add_argument('--n', type=int, default=5)

    # CUDA Arguments
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--cuda_number', type=int, default=0)

    # Model Arguments
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=3)

    # Predict Arguments
    parser.add_argument('--given_start_date', '-gsd', type=lambda s: pd.to_datetime(s))
    parser.add_argument('--predict_start_date', '-psd', type=lambda s: pd.to_datetime(s))
    parser.add_argument('--predict_end_date', '-ped', type=lambda s: pd.to_datetime(s))

    return parser


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

    device = torch.device(f"cuda:{args.cuda_number}" if torch.cuda.is_available() and args.use_gpu else "cpu")

    model_path = f'./result/{code}_m_{m}_n_{n}_split_{":".join(args.split_ratio)}.pt'
    model = torch.load(model_path)
    model.device = device

    test_data_path = os.path.join(prepro_path, 'test', f'{code}.csv')

    test_data = pd.read_csv(test_data_path)
    test_data.Date = pd.to_datetime(test_data['Date'], infer_datetime_format=True)

    given_start_date = args.given_start_date
    predict_start_date = args.predict_start_date
    predict_end_date = args.predict_end_date

    data = test_data[(test_data.Date >= given_start_date) & (test_data.Date <= predict_end_date)]

    first = torch.FloatTensor(data.iloc[0].to_numpy()[1:-1].astype(np.float64))

    x = torch.FloatTensor(data[data.Date < predict_start_date].to_numpy()[:, 1:-1].astype(np.float64))
    y = torch.FloatTensor(data.to_numpy()[:, 4].astype(np.float64))

    # normalize data
    x = x / first

    m = len(x)
    n = len(data) - m

    x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        out, _, _ = model(x, m, n)
        out = out.squeeze()

    close_price = out[:, 4] * first[4]

    x = list(range(1, m + n + 1))

    sns.lineplot(x=x, y=y.numpy())
    sns.lineplot(x=x, y=close_price.cpu().numpy())
    plt.legend(loc='upper left', labels=['true', 'pred'])

    plt.xlabel('Date')
    plt.ylabel('Close')

    plt.savefig(f'./result/{code}_m_{m}_n_{n}_split_{":".join(args.split_ratio)}_{given_start_date.strftime("%Y%m%d")}'
                f'_{predict_start_date.strftime("%Y%m%d")}_{predict_end_date.strftime("%Y%m%d")}.png')
