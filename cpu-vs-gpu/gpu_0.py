# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
# https://stackoverflow.com/questions/58002600/how-do-sessions-and-parallelism-work-in-tf2-0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import math as m
import progressbar as pb
import tensorflow as tf
import utils as u
from datetime import datetime as dt

# --- script settings start ---
document = '../data/marktwain.txt'
yml = './simple_rnn.yml'
# --- script settings end ---

FLAGS =  u.load_flags(yml)
tokens = u.load_tokens(document)
unique_tokens = set(tokens)
unique_tokens = dict((v, i) for i, v in enumerate(unique_tokens))
samples = u.make_samples(tokens, FLAGS['sample_length'])

# build a simple model
# words -> one-hot -> rnn -> dense -> output
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN( units = FLAGS['units'], input_shape = (FLAGS['sample_length'] - 1, len(unique_tokens))),
    tf.keras.layers.Dense(len(unique_tokens)),
    tf.keras.layers.Activation('softmax')])
optimizer = tf.keras.optimizers.Nadam()
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy')

t1 = dt.now()
sz = samples.shape
for i in range(0, FLAGS['epochs']):
    widgets = [ f'Epoch {i+1} batch ', pb.Counter(format = '%(value)d/%(max_value)d'), ' ', pb.Bar(marker = '.', left = '[', right = ']'), ' ', pb.ETA() ]
    with pb.ProgressBar(widgets = widgets, max_value = m.ceil(sz[0]/FLAGS['batch_size'])) as bar:
        batch_i = 0        
        while batch_i < sz[0]:
            bar.update(batch_i/FLAGS['batch_size'])
            batch = u.make_batch(samples, batch_i, FLAGS['batch_size'])
            one_hot = u.one_hot_batch(batch, unique_tokens)
            loss = model.train_on_batch(one_hot.x, one_hot.y)
            batch_i = batch_i + FLAGS['batch_size']
    print(f'Epoch: {i+1}, Loss: {loss}')

t2 = dt.now()
print(t2 - t1)
