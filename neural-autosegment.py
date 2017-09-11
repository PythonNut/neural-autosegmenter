import re
import random
import itertools as it
import numpy as np
import warnings
import keras
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

def get_text():
    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    with open(path) as f:
        text = f.read()
    text = re.sub(r'\s+', ' ', text)
    return text

def prepare_sequences(text, chunklen):
    sequences = []
    start_index = 0
    scan_index = 0
    chunk_index = 0
    sequence = [None] * chunklen
    spaces = [False] * chunklen
    while True:
        if scan_index == len(text):
            break
        if chunk_index == chunklen:
            sequences.append((sequence, spaces))
            chunk_index = 0
            start_index += 1
            scan_index = start_index
            sequence = [None] * chunklen
            spaces = [False] * chunklen

        if text[scan_index] == ' ':
            if chunk_index < chunklen:
                spaces[chunk_index] = True
            scan_index += 1

        else:
            sequence[chunk_index] = text[scan_index]
            scan_index += 1
            chunk_index += 1
    return sequences

def get_char_maps(text):
    chars = sorted(list(set(text)))
    char_index = {c: i for i, c in enumerate(chars)}
    index_char = {i: c for i, c in enumerate(chars)}
    return char_index, index_char


def vectorize_sequences(sequences, char_index_map, chunklen):
    X = np.zeros((len(sequences), chunklen, len(char_index_map)), dtype=np.bool)
    Y = np.zeros((len(sequences), chunklen), dtype=np.bool)
    for seq_idx, (sequence, spaces) in enumerate(sequences):
        for i, char in enumerate(sequence):
            X[seq_idx, i, char_index_map[char]] = True
        np.copyto(Y[seq_idx], spaces)
    return X, Y

def build_model(chunklen, char_map):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(chunklen, len(char_map))))
    model.add(LSTM(128))
    model.add(Dense(chunklen, activation='sigmoid'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train(model, X, Y, callbacks):
    return model.fit(
        X,
        Y,
        batch_size=128,
        epochs=100000,
        validation_split=0.33,
        callbacks=callbacks
    )

def plot_model(model, sequences, char_index_map, chunklen, axes):
    dim=4
    for i, j in it.product(*[range(dim)]*2):
        sequence = random.choice(sequences)
        x, y = vectorize_sequences([sequence], char_index_map, 40)
        p = model.predict(x)
        print('Cell: ({}, {}) '.format(i, j), end='')
        s = ''.join(sequence[0])
        for idx, c in enumerate(s):
            if p[0][idx] > 0.5:
                print(' ', end='')
            print(c, end='')
        print()
        axes[i, j].cla()
        axes[i, j].plot(y[0]*1.1)
        axes[i, j].plot(p[0])
    plt.draw()
    plt.pause(0.01)

def plot_loss(history, axes):
    # summarize history for loss
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].title('model loss')
    axes[1].ylabel('loss')
    axes[1].xlabel('epoch')
    axes[1].legend(['train', 'test'], loc='upper left')

class PlotModel(keras.callbacks.Callback):
    def __init__(self, sequences, char_index_map, chunklen):
        super().__init__()
        self.sequences = sequences
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
        if batch % 20 == 0:
            self.fig.canvas.draw()
            plt.pause(0.01)

    def on_epoch_end(self, epoch, logs=None):
        plot_model(
            self.model,
            self.sequences,
            self.char_map,
            self.chunklen,
            self.axes
        )
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
            ncols=2,
            sharex=True,
            sharey=True
        )

        self.fig.suptitle('Loss vs epoch')
        plt.draw()
        plt.pause(0.1)

    def on_batch_end(self, batch, logs):
        if batch % 20 == 0:
            self.fig.canvas.draw()
            plt.pause(0.01)

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

        self.axes[0].cla()
        self.axes[0].plot(self.x, self.losses, label="loss")
        self.axes[0].plot(self.x, self.val_losses, label="val_loss")
        self.axes[0].legend()

        self.axes[1].cla()
        self.axes[1].plot(self.x, self.acc, label="acc")
        self.axes[1].plot(self.x, self.val_acc, label="val_acc")
        self.axes[1].legend()

        self.fig.canvas.draw()
        plt.pause(0.01)

def main():
    chunklen = 40
    text = get_text()
    print('Preparing sequences')
    sequences = prepare_sequences(text, chunklen)
    char_index_map, index_char_map = get_char_maps(text)
    print('Vectorizing sequences')
    X, Y = vectorize_sequences(sequences, char_index_map, chunklen)
    model = build_model(chunklen, char_index_map)

    checkpointer = ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True)
    modelplotter = PlotModel(sequences, char_index_map, chunklen)
    lossplotter = PlotLoss()

    train(model, X, Y, [checkpointer, modelplotter, lossplotter])

if __name__ == '__main__':
    pass
    # main()
