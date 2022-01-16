import json, sys
import numpy as np
import pandas as pd
import random

from tensorflow.keras.models import load_model
from clustering_btc_price import output_jpg


def parse_price_to_change(prices):
    _max_length = 672
    _open = prices[0]
    list_prices_ratio_changes = [int((_p - _open) / _open * 100000) / 1000 for _p in prices]
    return list_prices_ratio_changes[:_max_length]


def parse_change_to_price(changes, basic_price):
    _rate = 100 / 16
    return [(_c / _rate * basic_price) + basic_price for _c in changes]


if __name__ == '__main__':

    argvs = sys.argv[1:]
    length_argvs = len(argvs)

    csv_path = 'csv/t1.csv'

    if length_argvs == 1:

        csv_path = argvs[0]
    

    lstm_model = load_model('projectbtc/model/_saved_lstm.h5')

    df_csv_data = pd.read_csv(csv_path)

    open_next_price = df_csv_data['Close'].iloc[-1]
    closes = np.array(parse_price_to_change(df_csv_data['Close']))
    # print('closes shape: ', closes.shape)
    closes = np.reshape(closes, (closes.shape[0], 1))

    y_pred = lstm_model.predict([closes])[0]

    # print('y_pred shape: ', y_pred.shape)
    # print(y_pred)

    # output_jpg([_[0] for _ in closes], name='predict_x')
    # output_jpg(y_pred, name='predict_result')

    pred_prices = parse_change_to_price(y_pred, open_next_price)
    df_csv = pd.DataFrame(pred_prices, columns=['Price'])
    df_csv.to_csv('output/predict_{}'.format(csv_path[-13:]))

    output_jpg(pred_prices, name='predict_price')

    print('Maximal Of pred_prices: ', max(pred_prices))
    print('Minimal Of pred_prices: ', min(pred_prices))