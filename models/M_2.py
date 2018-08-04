from __future__ import print_function

import pandas as pd
import numpy as np
from heraspy.callback import HeraCallback

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

from sklearn.cross_validation import train_test_split

herasCallback = HeraCallback(
    'LSTM_with_dropout_2',
    'localhost',
    4000
)

num_chars = 31
maxlen = 400
batch_size = 128
_nb_epoch = 500

alphabet = "abcdefghijklmnopqrstuvwxyz@.!?"
char_dict = dict((c, i) for i, c in enumerate(alphabet))
def char_to_int(x):
    #return ord(x) % num_chars
    if x in char_dict:
        return char_dict[x]
    return 30

def to_one_hot_char(s):
    l = []
    s = s.lower()
    for char in s:
        l.append(char_to_int(char))
    if len(l) < maxlen:
        l += [(ord(' ') % num_chars)]* (maxlen - len(l))
        
    del l[maxlen:]
    tmp = np.asarray(l)
    return np.eye(num_chars)[tmp]
df = pd.read_csv('train.csv')

print('Loading data...')
X_ = df['body']
X_ = np.array(X_.apply(to_one_hot_char))
X_ = np.dstack(X_)
X_= np.rollaxis(X_,-1)

print(X_.shape)
Y_ = df['is_harassment']

X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.2, random_state=0)

print('Loading data...')
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Pad sequences (samples x time)')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Build model...')
model = Sequential()
#model = load_model('model_weights_10_12.h5')
model.add(LSTM(256,input_shape=(maxlen,num_chars), dropout_W=0.3, dropout_U=0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy','matthews_correlation'])

print('Train...')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=_nb_epoch,
          validation_data=(X_test, Y_test),verbose=1,class_weight={0:1, 1:2})
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
Y_pred = model.predict_classes(X_test)
print('Test score:', score)
print('Test accuracy:', acc)
from sklearn.metrics import f1_score
print(f1_score(Y_test, Y_pred,average=None))
model.save('model_weights_11_12_2.h5') 
