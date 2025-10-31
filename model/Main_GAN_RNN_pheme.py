# Main_GAN_RNN_pheme.py

import sys
import os
import numpy as np
from numpy.testing import assert_array_almost_equal
import time
import datetime
import random

from model_GAN_RNN import GAN
from train import *
from evaluate import *
from Util import *


## variable ##
vocabulary_size = 5000
hidden_dim = 100
Nclass = 2
Nepoch = 10  # main epoch
Nepoch_G = 51  # pre-Train G
Nepoch_D = 101  # pre-Train D

lr_g = 0.005
lr_d = 0.005

obj = "pheme"  # dataset
fold = "0"

unit = f"GAN-RNN-{obj}{fold}"
modelPath = f"../param/param-{unit}.npz"

unit_dis = f"RNN-{obj}{fold}"
modelPath_dis = f"../param/param-{unit_dis}.npz"

unit_pre = f"GAN-RNN-pre-{obj}{fold}"
modelPath_pre = f"../param/param-{unit_pre}.npz"

trainPath = f"../nfold/TrainSet_{obj}{fold}.txt"
testPath = f"../nfold/TestSet_{obj}{fold}.txt"
labelPath = f"../resource/{obj}-label_balance.txt"
textPath = f"../resource/{obj}.vol_5000.txt"

################################### tools #####################################
def dic2matrix(dicW):
    # format: dicW = {ts:[index:wordfreq]}
    X = []
    timestamps = sorted(dicW.keys())
    for ts in timestamps:
        x = [0 for _ in range(vocabulary_size)]
        for pair in dicW[ts]:
            idx, val = pair.split(':')
            x[int(idx)] = int(val)
        X.append(x)
    return X


labelset_true = ['true', 'non-rumour']
labelset_false = ['false', 'rumour']


def loadLabel(label):
    if label in labelset_true:
        y_train = [1, 0]
        y_train_gen = [0, 1]
    elif label in labelset_false:
        y_train = [0, 1]
        y_train_gen = [1, 0]
    else:
        raise ValueError(f"Unknown label: {label}")
    return y_train, y_train_gen


################################# load data ###################################
def loadData():
    print("loading labels", end=" ")
    labelDic = {}
    with open(labelPath, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            eid, label = parts[0], parts[1]
            labelDic[eid] = label
    print(len(labelDic))

    print("reading events", end=" ")
    textDic = {}
    with open(textPath, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            eid, ts, Vec = parts[0], int(parts[1]), parts[2].split(' ')
            if eid in textDic:
                textDic[eid][ts] = Vec
            else:
                textDic[eid] = {ts: Vec}
    print(len(textDic))

    print("loading train set", end=" ")
    x_word_train, y_train, y_gen_train, Len_train = [], [], [], []
    index_true, index_false = [], []
    c = 0

    with open(trainPath, encoding='utf-8') as f:
        for eid in f:
            eid = eid.rstrip()
            if eid not in labelDic or eid not in textDic:
                continue
            label = labelDic[eid]
            if label in labelset_true:
                index_true.append(c)
            if label in labelset_false:
                index_false.append(c)

            y, y_gen = loadLabel(label)
            y_train.append(y)
            y_gen_train.append(y_gen)
            Len_train.append(len(textDic[eid]))
            wordFreq = dic2matrix(textDic[eid])
            x_word_train.append(wordFreq)
            c += 1
    print(c)

    print("loading test set", end=" ")
    x_word_test, y_test, Len_test = [], [], []
    c = 0
    with open(testPath, encoding='utf-8') as f:
        for eid in f:
            eid = eid.rstrip()
            if eid not in labelDic or eid not in textDic:
                continue
            label = labelDic[eid]
            y, y_gen = loadLabel(label)
            y_test.append(y)
            Len_test.append(len(textDic[eid]))
            wordFreq = dic2matrix(textDic[eid])
            x_word_test.append(wordFreq)
            c += 1
    print(c)

    return x_word_train, y_train, y_gen_train, Len_train, x_word_test, y_test, index_true, index_false


##################################### MAIN ####################################
if __name__ == "__main__":
    # 1. load tree & word & index & label
    x_word_train, y_train, yg_train, Len_train, x_word_test, y_test, index_true, index_false = loadData()

    # 2. init RNN model
    t0 = time.time()
    GANmodel = GAN(vocabulary_size, hidden_dim, Nclass)
    t1 = time.time()
    print(f"GAN-RNN model established, {(t1 - t0) / 60:.2f} min")

    # 3. pre-train or load model
    if os.path.isfile(modelPath):
        GANmodel = load_model(modelPath, GANmodel)
        lr_d, lr_g = 0.0001, 0.0001
    else:
        # pre-train classifier (Discriminator)
        if os.path.isfile(modelPath_dis):
            GANmodel = load_model_dis(modelPath_dis, GANmodel)
        else:
            pre_train_Discriminator(GANmodel, x_word_train, y_train, x_word_test, y_test, lr_d, Nepoch_D, modelPath_dis)

        # pre-train generator
        if os.path.isfile(modelPath_pre):
            GANmodel = load_model(modelPath_pre, GANmodel)
        else:
            pre_train_Generator('nr', GANmodel, x_word_train, index_true, Len_train, y_train, yg_train, lr_g, Nepoch_G, modelPath_pre)
            pre_train_Generator('rn', GANmodel, x_word_train, index_false, Len_train, y_train, yg_train, lr_g, Nepoch_G, modelPath_pre)

    # 4. train both Generator and Discriminator jointly
    train_Gen_Dis(
        GANmodel,
        x_word_train,
        Len_train,
        y_train,
        yg_train,
        index_true,
        index_false,
        x_word_test,
        y_test,
        lr_g,
        lr_d,
        Nepoch,
        modelPath
    )
