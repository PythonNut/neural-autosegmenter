import os
import re
import random
import itertools as it
import numpy as np
import warnings
import keras
import shutil

import operator
import string

from functools import reduce
from unidecode import unidecode

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.regularizers import L1L2
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint

import matplotlib as mpl
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore",".*GUI is implemented.*")

def get_char_maps(chars):
    char_index = {c: i for i, c in enumerate(chars)}
    index_char = {i: c for i, c in enumerate(chars)}
    return char_index, index_char

def prepare_sequences(text, start_index, chunklen):
    # sequences = []
    # start_index = 0
    scan_index = start_index
    chunk_index = 0
    sequence = [None] * chunklen
    spaces = [False] * chunklen
    if text[start_index] == ' ':
        return None
    if start_index > 0 and text[start_index - 1] == ' ':
        spaces[0] = True
    while True:
        if scan_index == len(text):
            break
        if chunk_index == chunklen:
            return sequence, spaces
            # sequences.append((sequence, spaces))
            # chunk_index = 0
            # start_index += 5
            # scan_index = start_index
            # sequence = [None] * chunklen
            # spaces = [False] * chunklen

        if text[scan_index] == ' ':
            if chunk_index < chunklen:
                spaces[chunk_index] = True
                scan_index += 1

        else:
            sequence[chunk_index] = text[scan_index]
            scan_index += 1
            chunk_index += 1
    return None
    # return sequences

def vectorize_sequences(sequences, char_index_map, chunklen):
    sequences = [x for x in sequences if x]
    X = np.zeros((len(sequences), chunklen, len(char_index_map)), dtype=np.bool)
    Y = np.zeros((len(sequences), chunklen), dtype=np.bool)
    for seq_idx, (sequence, spaces) in enumerate(sequences):
        for i, char in enumerate(sequence):
            X[seq_idx, i, char_index_map[char]] = True
            np.copyto(Y[seq_idx], spaces)
    return X, Y

def get_texts():
    dirs = os.listdir('corpus')
    texts = []

    terminal_width = shutil.get_terminal_size((80, 24)).columns
    print('Loading corpus')
    for d in dirs:
        # if random.random() > 0.1: continue
        print('Loading {}'.format(d).ljust(terminal_width), end='\r')
        with open('corpus/' + d) as f:
            try:
                texts.append(re.sub(r'\s+', ' ', unidecode(f.read())).strip())
            except UnicodeDecodeError: pass

    print('Collecting characters'.ljust(terminal_width))
    raw_chars = reduce(operator.__or__, map(set,texts))
    chars = list(sorted(set(x for x in raw_chars if x not in string.whitespace)))

    char_index_map, index_char_map = get_char_maps(chars)
    return ' '.join(texts), chars, char_index_map, index_char_map

def generate_training_data(text, char_index_map, chunklen):
    # blocksize = chunklen * 2

    # while True:
    #     try:
    #         text = random.choices(texts,weights=list(map(len,texts)))[0]
    #         block = random.randint(0, len(text)//blocksize - 2)
    #         offset = random.randint(0, blocksize)
    #     except ValueError:
    #         continue
    #     block_text = text[block*blocksize  +offset:(block + 1)*blocksize + offset]
    #     sequences = prepare_sequences(block_text, chunklen)
    #     sequences = sequences[:1]
    #     X, Y = vectorize_sequences(sequences, char_index_map, chunklen)
    #     yield X, Y
    batch = []
    batchsize = 64
    while True:
        print('Shuffling dataset')
        indexes = np.arange(len(text))
        np.random.shuffle(indexes)
        for i in indexes:
            seq = prepare_sequences(text, i, chunklen)
            if not seq: continue
            batch.append(seq)
            if len(batch) == batchsize:
                X, Y = vectorize_sequences(batch, char_index_map, chunklen)
                yield X, Y
                batch = []

def build_model(chunklen, char_map):
    model = Sequential()
    # model.add(LSTM(128, return_sequences=True, input_shape=(chunklen, len(char_map))))
    model.add(LSTM(128, input_shape=(chunklen, len(char_map))))
    # model.add(LSTM(128))
    model.add(Dense(chunklen, activation='sigmoid'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train(model, G, callbacks, chunklen):
    return model.fit_generator(
        G,
        validation_data=G,
        steps_per_epoch=10000,
        epochs=100000,
        validation_steps=100,
        callbacks=callbacks
    )

class PlotModel(keras.callbacks.Callback):
    def __init__(self, G, char_index_map, chunklen):
        super().__init__()
        self.data_generator = G
        self.char_map = char_index_map
        self.chunklen = chunklen
        self.fig, self.axes = plt.subplots(
            nrows=4,
            ncols=4,
            sharex=True,
            sharey=True
        )
        self.fig.suptitle("Model fit samples")
        plt.draw()
        plt.pause(0.1)

    def on_batch_end(self, batch, logs):
        if batch % 1000 == 0:
            self.fig.canvas.draw()
            # plt.pause(0.0001)

    def on_epoch_end(self, epoch, logs=None):
        dim=4
        for i, j in it.product(*[range(dim)]*2):
            X, Y = next(self.data_generator)
            X, Y = X[:1], Y[:1]
            p = self.model.predict(X)
            print('Cell: ({}, {}) '.format(i, j), end='')
            s = ''.join(self.char_map[x] for x in np.where(X[0]==True)[1])
            for idx, c in enumerate(s):
                if p[0][idx] > 0.5:
                    print(' ', end='')
                print(c, end='')
            print()
            self.axes[i, j].cla()
            self.axes[i, j].plot(Y[0]*1.1)
            self.axes[i, j].plot(p[0])
        plt.draw()
        plt.pause(0.01)
        self.fig.canvas.draw()

class PlotLoss(keras.callbacks.Callback):
    def __init__(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []
        self.fig, self.axes = plt.subplots(
            nrows=1,
            ncols=2
        )

        self.fig.suptitle('Loss vs epoch')
        plt.draw()
        plt.pause(0.1)

    def on_batch_end(self, batch, logs):
        if batch % 1000 == 0:
            self.fig.canvas.draw()
            # plt.pause(0.0001)

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

        self.axes[0].cla()
        self.axes[0].plot(self.x, self.losses, '-o', label="loss")
        self.axes[0].plot(self.x, self.val_losses, '-o', label="val_loss")
        self.axes[0].legend()

        self.axes[1].cla()
        self.axes[1].plot(self.x, self.acc, '-o', label="acc")
        self.axes[1].plot(self.x, self.val_acc, '-o', label="val_acc")
        self.axes[1].legend()

        self.fig.canvas.draw()
        plt.pause(0.01)

def main():
    text, chars, char_index_map, index_char_map = get_texts()
    chunklen = 40
    model = build_model(chunklen, char_index_map)
    G = generate_training_data(text, char_index_map, chunklen)

    checkpointer = ModelCheckpoint(filepath='model.{epoch:03d}-{val_loss:.4f}.hdf5', save_best_only=False)
    modelplotter = PlotModel(G, index_char_map, chunklen)
    lossplotter = PlotLoss()

    train(model, G, [checkpointer, modelplotter, lossplotter], chunklen)

if __name__ == '__main__':
    pass
    # main()
