import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import math as m
import numpy as np
import pathlib
import progressbar as pb
import string
import yaml
from collections import namedtuple
from datetime import datetime as dt
from typeguard import typechecked

# --- script settings start ---
document = '../data/marktwain.txt'
yml = './simple_rnn.yml'
# --- script settings end ---

HERE = pathlib.Path(__file__).parent
xy_data = namedtuple('xy_data', 'x, y')

with open(HERE.joinpath(yml)) as file:
    FLAGS = yaml.load(file, Loader = yaml.SafeLoader)

with open(HERE.joinpath(document)) as file:
    lines = file.readlines()
    # ~~ tokenize_words(simplify = T)
    lines = [x.split() for x in lines]
    tokens = [x for y in lines for x in y]
    tokens = [x.lower() for x in tokens if x not in string.punctuation]
unique_tokens = set(tokens)
unique_tokens = dict((v, i) for i, v in enumerate(unique_tokens))

max_word_length = max([len(x) for x in unique_tokens])
samples = np.empty([len(tokens) - FLAGS['sample_length'] + 1, FLAGS['sample_length']], dtype = f'<U{max_word_length}')
for i in range(0, samples.shape[0]):
    for j in range(0, samples.shape[1]):
        samples[i,j] = tokens[i + j]

# build a simple model
# words -> one-hot -> rnn -> dense -> output
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(units = FLAGS['units']))
model.add(tf.keras.layers.Dense(len(unique_tokens)))
model.add(tf.keras.layers.Activation('softmax'))
model.compile(
    optimizer = tf.keras.optimizers.Nadam(),
    loss = 'categorical_crossentropy')

@typechecked
def make_batch(samples: np.array, start: int, batch_size: int) -> np.array:
    sz = samples.shape
    batch = samples[range(start, min(sz[0], start + batch_size))]
    batch_x = batch[:, range(0, sz[1]-1)]
    batch_y = batch[:, sz[1]-1]
    return xy_data(batch_x, batch_y)
@typechecked
def one_hot_batch(batch: xy_data, unique_tokens: dict) -> np.array:
    sz = batch.x.shape
    one_hot_x = np.zeros([sz[0], sz[1], len(unique_tokens)], dtype = 'float32')
    one_hot_y = np.zeros([len(batch.y), len(unique_tokens)], dtype = 'float32')
    for i in range(0, sz[0]):
        for j in range(0, sz[1]):
            one_hot_x[i, j, unique_tokens[batch.x[i, j]]] = True
    for i in range(0, len(batch.y)):
        one_hot_y[i, unique_tokens[batch.y[i]]] = True
    return xy_data(one_hot_x, one_hot_y)

t1 = dt.now()
sz = samples.shape
widgets = [ 'batch ', pb.Counter(format = '%(value)d/%(max_value)d'), ' ', pb.Bar(marker = '.', left = '[', right = ']'), ' ', pb.ETA() ]
for i in range(0, FLAGS['epochs']):
    with pb.ProgressBar(widgets = widgets, max_value = m.ceil(sz[0]/FLAGS['batch_size'])) as bar:
        batch_i = 0        
        while batch_i < sz[0]:
            bar.update(batch_i/FLAGS['batch_size'])
            batch = make_batch(samples, batch_i, FLAGS['batch_size'])
            one_hot = one_hot_batch(batch, unique_tokens)
            loss = model.train_on_batch(one_hot.x, one_hot.y)
            batch_i = batch_i + FLAGS['batch_size']
    print(f'Epoch: {i+1}, Loss: {loss}')

t2 = dt.now()
print(t2 - t1)
