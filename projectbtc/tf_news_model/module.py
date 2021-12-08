import io
import os
import re
import shutil
import string
import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization


from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import get_data_home

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase, '[%s]' , '')


# Vocabulary size and number of words in a sequence.
# vocab_size = 10000
# sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
# vectorize_layer = TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=vocab_size,
#     output_mode='int',
#     output_sequence_length=sequence_length)

if __name__ == '__main__':
    
    texts = ['hi', 'good', 'bad', 'damn', 'hello', 'nice', 'cool']
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    number_of_clusters = 5

    model = KMeans(n_clusters=number_of_clusters, 
                init='k-means++', 
                max_iter=20, # Maximum number of iterations of the k-means algorithm for a single run.
                n_init=1)  # Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

    model.fit(X)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(number_of_clusters):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :3]:
            print(' %s' % terms[ind])

