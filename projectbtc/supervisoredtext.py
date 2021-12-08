import json, os
import numpy as np


from unit.fileparse import openJSON


VOCABULARY_SIZE = 32767
AUTO_CLASSES = 10
SEQUENCE_LENGTH = 120

def get_news_json_data():
    _static_path = 'json/news-bitcoin-com-websites-20211110_1357.json'
    return openJSON(_static_path)

def get_bitcoin_json_data():
    _static_path = 'json/bitcion2012-2021.json'
    return openJSON(_static_path)

if __name__ == '__main__':

    json_website_data = get_news_json_data()
    json_bitcoin_data = get_bitcoin_json_data()

    print(len(json_website_data))
    print(json_website_data[1])
    
    