from __future__ import print_function

import numpy as np
import pandas as pd
import utils
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.callbacks import TensorBoard
from sklearn.metrics import f1_score
from keras.layers.convolutional import Convolution1D
import json
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name",required=True, help="Model Name")
parser.add_argument("-e", "--epochs", type=int, help="Number of Epochs")
parser.add_argument("--maxlen", type=int, default = 400, help="Length of the sequences")
parser.add_argument("--model_size", type=int, default=256, help="Amount of LSTM units")
parser.add_argument("--ratio", type=int, default=2, help="Ratio of harassing datapoint error")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used during training")
parser.add_argument("--dropout", default=0.2, help="Dropout used in training")
parser.add_argument("--alphabet", default='abcdefghijklmnopqrstuvwxyz@.!? ', help="Alphabet used for encoding the text")

args = parser.parse_args()
model_params = {'model_name' : args.name ,'model_size' : args.model_size,
    'epochs': args.epochs,'ratio': args.ratio,'batch_size': args.batch_size, 'dropout': args.dropout,
     'maxlen' :args.maxlen, 'alphabet' : args.alphabet}

utils.init(model_params)
X_train, X_test, Y_train, Y_test = utils.process_data('train.csv')

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Pad sequences (samples x time)')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Build model...')


def calculate_results(model,X_test,Y_test):
    results = {}
    Y_pred = model.predict_classes(X_test)
    loss, acc = model.evaluate(X_test, Y_test, batch_size=utils.model_params['batch_size'])
    f1_score_not_harassment, f1_score_harassment  = f1_score(Y_test, Y_pred,average=None)
    results['accuracy'] = acc
    results['loss'] = loss
    results['f1_not_harassment'] = f1_score_not_harassment
    results['f1_harassment'] = f1_score_harassment
    with open('models/' + utils.model_params['model_name'] + '_results.json', 'w') as fp:
        json.dump(results, fp)

    with open('models/' + utils.model_params['model_name'] + '_model.json', 'w') as fp:
        json.dump(model_params, fp)

    model.save('models/' + utils.model_params['model_name'] + '_weights.h5')

if os.path.isfile('models/' + utils.model_params['model_name'] + '_weights.h5'):
    model = load_model('models/' + utils.model_params['model_name'] + '_weights.h5')
    utils.model_params['model_name'] = utils.model_params['model_name'] + '_re'
    print('Loaded model')

else:
    model = utils.build_model(utils.model_params)
    print('New model')

print('Train...')

model.fit(X_train, Y_train,
    batch_size=utils.model_params['batch_size'], epochs=utils.model_params['epochs'],
    validation_data=(X_test, Y_test),verbose=1,class_weight={0:1, 1:utils.model_params['ratio']},
    callbacks=[TensorBoard(log_dir='./logs/'+  utils.model_params['model_name'], histogram_freq=0, write_graph=True, write_images=False)])

calculate_results(model,X_test,Y_test)
