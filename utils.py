import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv1D

from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras import backend as K


model_params = {}
num_chars = 0;
char_dict = {}
def init(params):
    global model_params
    global num_chars
    global char_dict
    model_params = params
    num_chars = len(model_params['alphabet'])+1
    char_dict = dict((c, i) for i, c in enumerate(model_params['alphabet']))

def process_data(filename):
    df = pd.read_csv(filename)
    df = shuffle(df)
    print(df['is_harassment'].describe())
    print('Loading data...')
    X_ = df['body'].values.tolist()
    print(len(X_))
    X2_ = []
    for i in range(0,200):
        X2_.append(to_one_hot_char(X_[i]))

    X2_ = np.stack(X2_,axis=0)
    print(X2_.shape)
    Y_ = df['is_harassment']
    Y_ = Y_[:200]
    return train_test_split(X2_, Y_, test_size=0.2, random_state=0)

def build_model(model_params):
    model = Sequential()
    model.add(Conv1D(32, 4, border_mode='same',input_shape=(model_params['maxlen'],num_chars)))
    model.add(LSTM(int(model_params['model_size']) ,go_backwards=True))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def char_to_int(x):
    if x in char_dict:
        return char_dict[x]
    return num_chars-1

def to_one_hot_char(s):
    l = []
    s = s.lower()
    for char in s:
        l.append(char_to_int(char))
    if len(l) < model_params['maxlen']:
        l += [num_chars-1]* (int(model_params['maxlen']) - len(l))

    del l[model_params['maxlen']:]
    #print([model_params['maxlen'],num_chars])
    sess = K.get_session()
    with sess.as_default():
        return (tf.reshape(tf.one_hot(l, num_chars),[model_params['maxlen'],num_chars])).eval();
