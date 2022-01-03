import json
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import SGD



def get_lstm_model():
    model = Sequential()

    # model.add(Input(shape=(672,)))
    
    model.add(LSTM(512, input_shape=(672, 2), return_sequences = True))

    model.add(Dropout(0.2))

    model.add(LSTM(256))

    model.add(Dense(168))
    
    model.compile(
        loss=Huber(),
        # loss="mean_squared_error",
        # loss="mae",
        # optimizer=SGD(lr=1e-2),
        optimizer="adam",
        metrics=["hinge"],
    )

    model.summary()

    return model


def get_custom_prices(dataframe):
    date_start = dataframe['Date'].iloc[0]
    date_end = dataframe['Date'].iloc[-1]
    _mean_close = dataframe['Close'].mean()
    # _mean_volume = dataframe['Volume'].mean()
    _list_prices = dataframe['Close'].tolist()
    # _list_volumes = dataframe['Volume'].tolist()

    high = dataframe['High'].max()
    low = dataframe['Low'].min()
    open = dataframe['Open'].iloc[0]
    close = dataframe['Close'].iloc[-1]

    list_prices_ratio_changes = [int((_p - open) / open * 100000) / 1000 for _p in _list_prices]

    return {
        'date_start': date_start,
        'date_end': date_end,
        'high': high,
        'low': low,
        'open': open,
        'close': close,
        'price_changes': list_prices_ratio_changes,
        'length_hour': len(list_prices_ratio_changes),
    }



def get_splited_point_dataframe(filepath, load_json=False):
    _size_y = 168
    _size_x = 168*4
    _json_file_path = 'json/tmp_save_btc_train_prices.json'
    xdata = []
    ydata = []

    if load_json:
        with open(_json_file_path, 'r') as f:
            res = json.load(f)
            xdata = res['x']
            ydata = res['y']
    else:
    
        df_btcusd = pd.read_csv(filepath)
        df_btcusd_sorted = df_btcusd.sort_values(by=['Unix Timestamp'])
        df_btcusd_sorted.reset_index(inplace=True, drop=True)

        _length_df = df_btcusd_sorted.shape[0]
        

        print(df_btcusd_sorted.head())

        print('_length_of_price_points: ', _length_df)

        print('  0.00%', end='\r')

        for idx, row in df_btcusd_sorted.iterrows():
            
            if idx >= (_length_df - (_size_x + _size_y)):
                break

            if row['Date'][-8:] == '00:00:00':  # is a day start
                _loc_x = df_btcusd_sorted.query('index > {} and index <= {}'.format(idx, idx + _size_x))
                _loc_y = df_btcusd_sorted.query('index >= {} and index < {}'.format(idx + _size_x, idx + _size_x + _size_y))
                
                _points_data = get_custom_prices(_loc_x)
                _ans = get_custom_prices(_loc_y)
                xdata.append(_points_data)
                ydata.append(_ans)

                _pr = idx / _length_df * 100
                print(' {:2.2f}%'.format(_pr), end='\r')
                
        print(' 100.00% - Done.')

        with open(_json_file_path, 'w') as f:
            json.dump({'x':xdata,'y':ydata}, f, indent=2)

    return xdata, ydata



if __name__ == '__main__':
    filepath = 'csv/gemini_BTCUSD_1hr.csv'

    xx, yy = get_splited_point_dataframe(filepath, load_json=True)
    # xx, yy = get_splited_point_dataframe(filepath, load_json=False)
    train_x = np.array([_x['price_changes'] for _x in xx])
    train_y = np.array([_y['price_changes'] for _y in yy])

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    print('train_y[Example][-10]: ', train_y[1200][-10:])

    lstm_model = get_lstm_model()

    history = lstm_model.fit(train_x, train_y, epochs=8)

    lstm_model.save('projectbtc/model/_saved_lstm.h5', save_format='h5')
    