import sys, os, time, datetime
import pandas as pd
import numpy as np
from clustering_btc_price import output_jpg
import matplotlib.pyplot as plt


def parse_to_ts(string):
    return int(time.mktime(datetime.datetime.strptime(_first_timestamp, "%Y-%m-%d %H:%M:%S").timetuple()) / (60*60))


if __name__ == '__main__':

    argvs = sys.argv[1:]
    length_argvs = len(argvs)

    csv_path = 'csv/20211129-0000.csv'
    subset_dir_path = 'results'

    if length_argvs == 1:

        csv_path = argvs[0]

    df_background_data = pd.read_csv(csv_path)
    df_subset_datas = []

    list_predicts = os.listdir(subset_dir_path)
    list_predicts.sort()
    
    print('list_predicts: ', list_predicts)

    for _file in list_predicts:
        _path = '{}/{}'.format(subset_dir_path, _file)
        if _path[-3:] == 'csv':
            _df_subset_data = pd.read_csv(_path)
            df_subset_datas.append(_df_subset_data)

    # fig = plt.figure(figsize=(20,16),dpi=100,linewidth = 1)
    # axe = fig.add_subplot(1, 1, 1)
    # plt.plot(month,stock_tsmcc,'s-',color = 'r', label="TSMC")
    ax = plt.gca()
    # df_background_data.plot(kind='line', x='Timestamp', y='Close', color='green', label='RealPrice', ax=ax)
    _timestamps = df_background_data['Timestamp']
    _first_timestamp = _timestamps[0]
    _closes = df_background_data['Close']
    plt.xlabel('start date: {}'.format(_first_timestamp), fontsize=14, labelpad = 2)
    plt.ylabel("price", fontsize=14, labelpad = 4)
    ax.plot(range(len(_timestamps)), _closes, label='RealPrice', color='green')
    # 2021-11-29 01:00:00
    _first_ts = parse_to_ts(_first_timestamp)
    _color_list = ['#ff0000', '#888800', '#ffff00', '#0000ff', '#880088', '#ff00ff', '#ff0000', '#888800', '#ffff00']
    _color_idx = 0

    for _ in df_subset_datas:
        _timestamps = _['Timestamp']
        _price = _['Price']
        _first_timestamp = _timestamps[0]
        _gap = parse_to_ts(_first_timestamp) - _first_ts 
        print('_gap hour: ', _gap)
        _ranges = range(_gap, _gap + 168)
        ax.plot(_ranges, _price, label='predict', color=_color_list[_color_idx])
        _color_idx += 1
        # _.plot(kind='line', x='Timestamp', y='Price', color='blue', ax=ax)
        
    # plt.show()
    plt.savefig('output/merged.png')

    # output_jpg([], name='merged')