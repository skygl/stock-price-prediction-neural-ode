from argparse import ArgumentParser
import os

import pandas as pd
from tqdm import tqdm


DAILY_PRICE_PATH = './data/daily_price'

YEAR_OF_START = 2011
MONTH_OF_START = 2
DAY_OF_START = 1

BASE_OUT_DIR = './prepro'


def get_ext_parser():
    parser = ArgumentParser()

    # train, valid, test split ratio which unit is "Y" from start date specified above.
    # i.e. "811" means that periods of train data, valid data and test data are 8Y, 1Y, and 1Y respectively.
    parser.add_argument('--split_ratio', type=str, default='811')

    return parser


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False


if __name__ == '__main__':
    parser = get_ext_parser()
    args = parser.parse_args()

    split_ratio = args.split_ratio

    split_ratio = list(map(int, list(split_ratio)))
    assert len(split_ratio) == 3
    train_period, valid_period, test_period = split_ratio

    out_path = os.path.join(BASE_OUT_DIR, f'split_{":".join(args.split_ratio)}')

    mkdir_if_not_exists(out_path)
    mkdir_if_not_exists(os.path.join(out_path, 'train'))
    mkdir_if_not_exists(os.path.join(out_path, 'valid'))
    mkdir_if_not_exists(os.path.join(out_path, 'test'))

    train_start_date = pd.Timestamp(YEAR_OF_START, MONTH_OF_START, DAY_OF_START)
    valid_start_date = pd.Timestamp(YEAR_OF_START + train_period, MONTH_OF_START, DAY_OF_START)
    test_start_date = pd.Timestamp(YEAR_OF_START + train_period + valid_period, MONTH_OF_START, DAY_OF_START)
    test_end_date = pd.Timestamp(YEAR_OF_START + train_period + valid_period + test_period, MONTH_OF_START, DAY_OF_START)

    # code_list = [_.split('.')[0] for _ in os.listdir(DAILY_PRICE_PATH) if _.endswith('.csv')]
    code_list = ['AAPL']

    train_data_list = []
    valid_data_list = []
    test_data_list = []

    for code in tqdm(code_list, desc='progress'):
        daily_price = pd.read_csv(f'{DAILY_PRICE_PATH}/{code}.csv')
        daily_price.Date = pd.to_datetime(daily_price['Date'], infer_datetime_format=True)
        daily_price = daily_price.reset_index(drop=True)

        train_daily_price = daily_price[(daily_price.Date >= train_start_date) & (daily_price.Date < valid_start_date)]
        valid_daily_price = daily_price[(daily_price.Date >= valid_start_date) & (daily_price.Date < test_start_date)]
        test_daily_price = daily_price[(daily_price.Date >= test_start_date) & (daily_price.Date < test_end_date)]

        train_result_path = os.path.join(out_path, 'train', f'{code}.csv')
        valid_result_path = os.path.join(out_path, 'valid', f'{code}.csv')
        test_result_path = os.path.join(out_path, 'test', f'{code}.csv')

        train_daily_price.to_csv(train_result_path, index=False)
        valid_daily_price.to_csv(valid_result_path, index=False)
        test_daily_price.to_csv(test_result_path, index=False)

