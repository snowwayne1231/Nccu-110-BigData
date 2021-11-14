import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, json
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset, save_time_series_txt, load_time_series_txt


def output_jpg(data, y='Y', x='X', name='example'):
    plt.plot(data)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.savefig('output/{}.jpg'.format(name))
    plt.figure().clear()


def get_tmp_dataframe():
    _static_path = 'json/tmp_price_points.json'
    jsondata = None
    with open(_static_path, 'r') as f:
        jsondata = json.load(f)
    return jsondata['results'], jsondata['answers']


def get_splited168point_dataframe(filepath):
    _data_length = 168
    df_btcusd = pd.read_csv(filepath)
    df_btcusd_sorted = df_btcusd.sort_values(by=['Unix Timestamp'])
    df_btcusd_sorted.reset_index(inplace=True, drop=True)

    _length_df = df_btcusd_sorted.shape[0]
    results = []
    answers = []

    print(df_btcusd_sorted.head())

    print('_length_of_price_points: ', _length_df)

    print('  0.00%', end='\r')

    for idx, row in df_btcusd_sorted.iterrows():
        
        if idx >= (_length_df - _data_length):
            break

        if row['Date'][-8:] == '00:00:00':  # is a day start
            _loc = df_btcusd_sorted.query('index > {} and index <= {}'.format(idx, idx+_data_length))
            _loc2 = df_btcusd_sorted.query('index >= {} and index < {}'.format(idx+_data_length, idx+_data_length*2))
            
            _points_data = get_custom_points(_loc)
            _ans = get_custom_answer(_loc2)
            results.append(_points_data)
            answers.append(_ans)

            _pr = idx / _length_df * 100
            print(' {:2.2f}%'.format(_pr), end='\r')
            

    print(' 100.00% - Done.')

    with open('json/tmp_price_points.json', 'w') as f:
        json.dump({'results':results,'answers':answers}, f, indent=2)

    return results, answers



def get_custom_points(dataframe):
    date = dataframe['Date'].iloc[-1]
    _mean_close = dataframe['Close'].mean()
    _mean_volume = dataframe['Volume'].mean()
    _list_prices = dataframe['Close'].tolist()
    _list_volumes = dataframe['Volume'].tolist()

    # define the points of price and volume
    _list_prices_ratio_changes = [int((_p - _mean_close) / _mean_close * 10000) / 100 for _p in _list_prices]
    # _list_volume_ratio_changes = [int((_v - _mean_volume) / _mean_volume * 10000) / 100 for _v in _list_volumes]

    # print(_list_prices_ratio_changes)
    # return {'date': date, 'pricepoints': _list_prices_ratio_changes, 'volumepoints': _list_volume_ratio_changes}
    return {'date': date, 'pricepoints': _list_prices_ratio_changes}


def get_custom_answer(dataframe):
    date = dataframe['Date'].iloc[0]
    price = dataframe['Close'].iloc[0]
    _list_hight = dataframe['High'].tolist()
    _list_low = dataframe['Low'].tolist()
    _max_hight = max(_list_hight)
    _min_low = min(_list_low)
    hieght_change = _max_hight / price
    low_change = _min_low / price
    return {'date': date, 'price': price, 'high': hieght_change, 'low': low_change}

def parse_time_series_data(listdata):
    formatted_time_series = to_time_series_dataset(listdata)

    print('formatted_time_series shape: ', formatted_time_series.shape)

    # save_time_series_txt('output/time_series.txt', formatted_time_series)

    return formatted_time_series




if __name__ == '__main__':

    argvs = sys.argv[1:]
    length_argvs = len(argvs)
    datapoints = []
    answers = []

    try:

        if length_argvs == 0:

            datapoints, answers = get_splited168point_dataframe('csv/gemini_BTCUSD_1hr.csv')
            
            # save_example_jpgs = [0, 100, -1]
            # for idx in save_example_jpgs:
            #     _loc = df_btc[idx]
            #     output_jpg(_loc['pricepoints'], y='Rate', x='Hours / Week Before [{}]'.format(_loc['date']), name=_loc['date'])


        elif length_argvs == 1:

            if argvs[0] == 'loaded':
                datapoints, answers = get_tmp_dataframe()
            else:
                print('argvs: ', argvs)
                exit(2)

            # formatted_time_series = load_time_series_txt('output/time_series.txt')

        print("Length of splited points: ", len(datapoints))

        train_x = [_['pricepoints'] for _ in datapoints]

        formatted_time_series = parse_time_series_data(train_x)
        np_answers = np.array(answers)
        # formatted_time_series = formatted_time_series[:20]
        # np_answers = np_answers[:20]
        
        km = TimeSeriesKMeans(n_clusters=5, metric="dtw", verbose=True)
        sz = formatted_time_series.shape[1]
        # km_model = km.fit(formatted_time_series)
        # print(km_model)

        y_pred = km.fit_predict(formatted_time_series)

        ans_results = []

        for yi in range(5):
            plt.subplot(3, 3, yi+1)
            target_list = y_pred == yi
            
            for xx in formatted_time_series[target_list]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            
            _answer_targeted = np_answers[target_list]
            _length_ans = len(_answer_targeted)
            _high = 0
            _low = 0
            for _at in _answer_targeted:
                _high += _at['high']
                _low += _at['low']
            _high = _high / _length_ans
            _low = _low / _length_ans

            ans_results.append({'high': _high, 'low': _low})

            plt.plot(km.cluster_centers_[yi].ravel(), "g-")
            plt.xlim(0, sz)
            plt.ylim(-10, 10)
            plt.text(0.55, 0.85,'Cluster {}'.format(yi + 1), transform=plt.gca().transAxes)

        #result
        plt.subplot(3, 3, 8)
        highs = [h['high'] for h in ans_results]
        lows = [l['low'] for l in ans_results]
        plt.plot(highs, "b-", alpha=1)
        plt.plot(lows, "r-", alpha=1)
        plt.xticks([0,1,2,3,4], ['C-1', 'C-2', 'C-3', 'C-4', 'C-5'])
        
        # plt.tight_layout()
        # plt.show()
        plt.savefig('output/result.jpg')
        
    except KeyboardInterrupt:

        print('Stop.')

    
        
    
    
    