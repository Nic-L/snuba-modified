

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn import *
from lstm.imdb_lstm import *

import matplotlib.pyplot as plt

dataset='imdb'

from data.loader import DataLoader
dl = DataLoader()
_, _, _, train_ground, val_ground, test_ground, train_text, val_text, test_text = dl.load_data(dataset=dataset)
train_reef = np.load('./data/imdb_reef.npy')

f1_all = []
pr_all = []
re_all = []
val_acc_all = []

bs_arr = [64, 128, 256]
n_epochs_arr = [5, 10, 25]

for bs in bs_arr:
    for n in n_epochs_arr:
        y_pred = lstm_simple(train_text, train_reef, val_text, val_ground, bs=bs, n=n)
        predictions = np.round(y_pred)

        val_acc_all.append(np.sum(predictions == val_ground) / float(np.shape(val_ground)[0]))
        f1_all.append(metrics.f1_score(val_ground, predictions))
        pr_all.append(metrics.precision_score(val_ground, predictions))
        re_all.append(metrics.recall_score(val_ground, predictions))

ii,jj = np.unravel_index(np.argmax(f1_all), (3,3))
print('Best Batch Size: ', bs_arr[ii])
print('Best Epochs: ', n_epochs_arr[jj])

print('Validation F1 Score: ', max(f1_all))
print('Validation Best Pr: ', pr_all[np.argmax(f1_all)])
print('Validation Best Re: ', re_all[np.argmax(f1_all)])

y_pred = lstm_simple(train_text, train_reef, test_text, test_ground, bs=bs_arr[ii], n=n_epochs_arr[jj])
predictions = np.round(y_pred)

print('Test F1 Score: ', metrics.f1_score(test_ground, predictions))
print('Test Precision: ', metrics.precision_score(test_ground, predictions))
print('Test Recall: ', metrics.recall_score(test_ground, predictions))