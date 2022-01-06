import json, sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, TextVectorization, Dropout
from unit.fileparse import openJSON
from sklearn.preprocessing import MinMaxScaler
from crawl_news.crawl import crawlBitcoinComWebsites, crawlNewsContentByJson


VOCABULARY_SIZE = 32767
AUTO_CLASSES = 10
SEQUENCE_LENGTH = 120


def get_news_to_btc_json_data():
    _static_path = 'json/news-to-btc.json'
    return openJSON(_static_path)


def get_np_dataset_content(dataset, content_idx):
    contents = [_[content_idx] for _ in dataset]
    return np.array(contents)


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

def get_news_model(np_content_ds):
    embedding_dim=8

    v_layer = get_vectorize_layer(VOCABULARY_SIZE, SEQUENCE_LENGTH)
    v_layer.adapt(np_content_ds)

    model = Sequential([
        v_layer,
        Embedding(VOCABULARY_SIZE, embedding_dim, name="Embedding"),
        GlobalAveragePooling1D(),
        Dense(embedding_dim, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

    return model


def parse_train_y(dataset, idx):
    next_data = [float(_[idx].replace('%', '')) for _ in dataset]

    np_next_data = np.array(next_data).reshape((-1,1))
    min_max_scaler = MinMaxScaler(feature_range=(0, 1)).fit(np_next_data)

    next_train_y = min_max_scaler.transform(np_next_data)

    np_next_data = next_train_y.reshape((-1))
    return np_next_data


if __name__ == '__main__':

    json_news_to_btc = get_news_to_btc_json_data()
    json_news_to_btc_data = json_news_to_btc['data']
    _idx_content = json_news_to_btc['columns'].index('content')
    _idx_change = json_news_to_btc['columns'].index('btc_change')
    
    np_content_ds = get_np_dataset_content(json_news_to_btc_data, _idx_content)

    train_y = parse_train_y(json_news_to_btc_data, _idx_change)
    
    model = get_news_model(np_content_ds)

    # print('train_y: ', train_y[1])
    # exit(2)
    
    argvs = sys.argv[1:]
    length_argvs = len(argvs)

    if length_argvs == 1 and argvs[0] == 'predict':

        model.load_weights('projectbtc/model/_saved_news_embedding')
        
        websites = crawlBitcoinComWebsites(5, 1)
        news_dataset = crawlNewsContentByJson(websites)
        
        _x = np.array([_['content'] for _ in news_dataset])
        
        _pred = model.predict([_x])[0]
        
        print('_pred:', _pred)
        
    else:

        model.fit(
            np_content_ds,
            train_y,
            epochs=50,
        )
        model.save('projectbtc/model/_saved_news_embedding')
        
    

    