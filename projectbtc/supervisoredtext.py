import json, os
import numpy as np
from datetime import datetime


from unit.fileparse import openJSON, saveJSON


VOCABULARY_SIZE = 32767
AUTO_CLASSES = 10
SEQUENCE_LENGTH = 120
DATE_FORMATE = "%Y/%m/%d"

def get_news_json_data():
    _static_path = 'json/news-bitcoin-com-websites-20211110_1357.json'
    return openJSON(_static_path)

def get_bitcoin_json_data():
    _static_path = 'json/bitcion2012-2021.json'
    return openJSON(_static_path)

def parse_text(str):
    return str.split(' ')

if __name__ == '__main__':

    json_website_data = get_news_json_data()
    json_bitcoin_data = get_bitcoin_json_data()

    print(json_website_data[1])
    # print(json_bitcoin_data['columns'])
    # ['Date', 'Price', 'Open', 'High', 'Low', 'Vol', 'Change%']
    print(json_bitcoin_data['data'][1])

    results = {
        'columns': ['date', 'title', 'content', 'type', 'btc_price', 'btc_open', 'btc_high', 'btc_low', 'btc_vol', 'btc_change'],
        'data': [],
    }

    btc_map = {}
    for _btcdata in json_bitcoin_data['data']:
        _date = _btcdata[0]
        btc_map[_date] = _btcdata
    

    for _wdata in json_website_data:
        _date = _wdata['date']
        _date = _date.replace(' ', '')
        _title = _wdata['title']
        _content = _wdata['content']
        _type = _wdata['type']

        btc_data = btc_map.get(_date, None)
        if btc_data is None:
            continue

        _next = [
            _date,
            _title,
            _content,
            _type,
            btc_data[1],
            btc_data[2],
            btc_data[3],
            btc_data[4],
            btc_data[5],
            btc_data[6],
        ]

        # _datetime = datetime.strptime(_date, DATE_FORMATE)
        results['data'].append(_next)

    saveJSON('news-to-btc', results)
    
    