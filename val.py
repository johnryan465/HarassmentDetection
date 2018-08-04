from __future__ import print_function

import pandas as pd
import numpy as np

from keras import backend as K
import tensorflow as tf
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
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.framework import graph_util
from tensorflow.contrib.session_bundle import session_bundle

np.set_printoptions(threshold=np.nan)

import json
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Model Name")
args = parser.parse_args()

with open("models/" + args.name + "_model.json") as data_file:
    model_params = json.load(data_file)
utils.init(model_params)

def process_data(filename):
    df_ = pd.read_csv(filename)
    df = shuffle(df_)
    print(df['is_harassment'].describe())
    print('Loading data...')

    for index, row in df.iterrows():
        text = row['body']
        test_ = np.expand_dims( utils.to_one_hot_char(text), axis=0)
        prediction = model.predict_classes(test_,verbose=0)[0][0]
        if prediction != row['is_harassment']:
            print(row['body'] , " correct answer ",row['is_harassment'])


def calculate_results(model,X_test,Y_test):
    results = {}
    Y_pred = model.predict_classes(X_test)
    loss, acc = model.evaluate(X_test, Y_test, batch_size=utils.model_params['batch_size'])
    f1_score_not_harassment, f1_score_harassment  = f1_score(Y_test, Y_pred,average=None)
    results['accuracy'] = acc
    results['loss'] = loss
    results['f1_not_harassment'] = f1_score_not_harassment
    results['f1_harassment'] = f1_score_harassment
    print(results)

if os.path.isfile('models/' + model_params['model_name'] + '_weights.h5'):
    model = load_model('models/' + model_params['model_name'] + '_weights.h5')
    model_params['model_name'] = model_params['model_name'] + '_re'
    print('Loaded model')
    while(True):
        s = raw_input()
        d = utils.to_one_hot_char(s)
        d = tf.reshape(d,[1,400,32])
        print(d)
        sess = K.get_session()
        with sess.as_default():
            print(d.eval())
            print(model.predict(d.eval()))
        #123456789
    #model.save("final.h5")
    #saver = tf.train.Saver()
    #saver.save(K.get_session(), '/tmp/keras_model.ckpt')
    #print([node.op.name for node in model.outputs])

    #print("saved")
    #X_train, X_test, Y_train, Y_test = utils.process_data('train.csv')
    #calculate_results(model,X_test,Y_test)
    #process_data('train.csv')
else:
    print('Model does not exist')
