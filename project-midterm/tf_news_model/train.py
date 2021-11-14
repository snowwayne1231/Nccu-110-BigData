import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, TextVectorization, Dropout


VOCABULARY_SIZE = 32767
AUTO_CLASSES = 10
SEQUENCE_LENGTH = 120



def get_news_json_data():
    _static_path = 'json/news-bitcoin-com-websites-20211110_1357.json'
    res = None
    with open(_static_path, 'r') as f:
        res = json.load(f)
    return res

def get_bitcoin_json_data():
    _static_path = 'json/bitcion2012-2021.json'
    res = None
    with open(_static_path, 'r') as f:
        res = json.load(f)
    return res


def get_vectorize_layer(size, length):

    def custom_standardization(input_text):
        lowercase = tf.strings.lower(input_text)
        return tf.strings.regex_replace(lowercase, '[%s]' , '')

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=size,
        output_mode='int',
        output_sequence_length=length)

    return vectorize_layer


def parse_y_map(data_list):
    _map = {}
    for _ in data_list:
        _date = _[0]
        _change = _[6]
        _volume = _[5]
        _year = int(_date[:4])
        if _year >= 2017:
            _map[_date] = [_change, _volume]
    
    return _map


if __name__ == '__main__':

    json_website_data = get_news_json_data()
    json_bitcoin_data = get_bitcoin_json_data()
    
    contents = [_['content'] for _ in json_website_data]
    np_content_ds = np.array(contents)

    v_layer = get_vectorize_layer(VOCABULARY_SIZE, SEQUENCE_LENGTH)
    v_layer.adapt(np_content_ds)

    embedding_dim=16

    train_y = parse_train_y(json_bitcoin_data['data'])

    model = Sequential([
        v_layer,
        Embedding(VOCABULARY_SIZE, embedding_dim, name="Embedding"),
        GlobalAveragePooling1D(),
        Dense(embedding_dim, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    