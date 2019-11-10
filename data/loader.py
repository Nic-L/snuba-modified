import numpy as np
import scipy
import json
import sklearn.model_selection
import pandas as pd
from sklearn.datasets import make_multilabel_classification

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

def parse_file(filename):

    def parse(filename):
        movies = []
        with open(filename) as f:
            for line in f:
                obj = json.loads(line)
                movies.append(obj)
        return movies

    f = parse(filename)
    gt = []
    plots = []
    idx = []
    for i,movie in enumerate(f):
        genre = movie['Genre']
        if 'Action' in genre and 'Romance' in genre:
            #continue
            plots = plots + [movie['Plot']]
            gt.append(1)
            idx.append(i)
        elif 'Action' in genre:
            plots = plots+[movie['Plot']]
            gt.append(2)
            idx.append(i)
        elif ('Romance' in genre) or ('Horror' in genre):
            plots = plots+[movie['Plot']]
            gt.append(3)
            idx.append(i)
        else:
            #continue
            plots = plots + [movie['Plot']]
            gt.append(4)
            idx.append(i)
    
    return np.array(plots), np.array(gt)

def split_data(X, plots, y):
    np.random.seed(1234)
    num_sample = np.shape(X)[0]
    num_test = 500

    X_test = X[0:num_test,:]
    X_train = X[num_test:, :]
    plots_train = plots[num_test:]
    plots_test = plots[0:num_test]

    y_test = y[0:num_test]
    y_train = y[num_test:]

    # split dev/test
    test_ratio = 0.2
    X_tr, X_te, y_tr, y_te, plots_tr, plots_te = \
        sklearn.model_selection.train_test_split(X_train, y_train, plots_train, test_size = test_ratio)

    return np.array(X_tr.todense()), np.array(X_te.todense()), np.array(X_test.todense()), \
        np.array(y_tr), np.array(y_te), np.array(y_test), plots_tr, plots_te, plots_test

def convert_to_numeric_label(labels1, labels2, labels3):
    labels_long = np.append(labels1, np.append(labels2, labels3))
    labels1_con = np.zeros(np.shape(labels1))
    labels2_con = np.zeros(np.shape(labels2))
    labels3_con = np.zeros(np.shape(labels3))

    labels_un = np.unique(labels_long)

    for counter, label in enumerate(labels_un):
       #counter + 1 um die 0 für abstain frei zu halten
       labels1_con[labels1 == label] = counter + 1
       labels2_con[labels2 == label] = counter + 1
       labels3_con[labels3 == label] = counter + 1

    return np.transpose(labels1_con.astype(np.int32)), np.transpose(labels2_con.astype(np.int32)), np.transpose(labels3_con.astype(np.int32))

def load_csv(path):
    skip_list = ["label", "file_name", "corpus", "sheet_name", "sheet_index", "table_name", "cell_address",
                 "first_row_num", "first_col_num"
        , "last_row_num", "last_col_num"]

    df = pd.read_csv(path, usecols=lambda column: column not in skip_list)
    ground_df = pd.read_csv(path, usecols=lambda column: column in ["label"])

    primitives = df.to_numpy()
    ground = ground_df.to_numpy()

    primitives = np.transpose(primitives)
    ground = np.transpose(ground)

    return primitives, ground

class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.01):
        val_sum = np.sum(np.abs(val_primitive_matrix),axis=0)
        train_sum = np.sum(np.abs(train_primitive_matrix),axis=0)

        #Only select the indices that fire more than 1% for both datasets
        train_idx = np.where((train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
        val_idx = np.where((val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
        common_idx = list(set(train_idx) & set(val_idx))

        return common_idx

    def load_data(self, dataset, data_path='./data/imdb/'):
        #Parse Files
        plots, labels = parse_file(data_path+'budgetandactors.txt')
        #read_plots('imdb_plots.tsv')

        #Featurize Plots
        vectorizer = CountVectorizer(min_df=1, binary=True, \
            decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        X = vectorizer.fit_transform(plots)
        valid_feats = np.where(np.sum(X,0)> 2)[1]
        X = X[:,valid_feats]

        #Split Dataset into Train, Val, Test
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_ground, val_ground, test_ground, \
            train_plots, val_plots, test_plots = split_data(X, plots, labels)

        #Prune Feature Space
        common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
        return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx], \
            np.array(train_ground), np.array(val_ground), np.array(test_ground), \
            train_plots, val_plots, test_plots

    def load_data_synt(self):

        #featureset, labelset = sklearn.datasets.make_multilabel_classification(n_samples=4000, n_features=30,
                                                                              # n_classes=2, n_labels=1)

        featureset, labelset = sklearn.datasets.make_classification(n_samples = 4000, n_features = 3, n_classes = 4, n_informative=3, n_redundant=0, n_repeated=0)

        labelset = labelset + 1

        #featureset[featureset >= 1] = 1.
        '''true_labels = []

        for row in labelset:
            if (row[0] == 1) and (row[1] == 1):
                true_labels.append(1)
            if (row[0] == 1) and (row[1] == 0):
                true_labels.append(2)
            if (row[0] == 0) and (row[1] == 1):
                true_labels.append(3)
            if (row[0] == 0) and (row[1] == 0):
                true_labels.append(4)

        true_labels = np.array(true_labels)'''

        X_train = featureset[500:, :]
        X_test= featureset[0:500, :]

        y_train = labelset[500:]
        y_test = labelset[0:500]

        test_ratio = 0.2
        X_tr, X_te, y_tr, y_te = \
            sklearn.model_selection.train_test_split(X_train, y_train, test_size=test_ratio)

        #Prune Feature Space
        common_idx = self.prune_features(np.array(X_te), np.array(X_tr))
        return np.array(X_tr)[:,common_idx], np.array(X_te)[:,common_idx], X_test[:,common_idx], \
            np.array(y_tr), np.array(y_te), np.array(y_test)

    def load_data_sheet(self):

        train_primitives, train_ground = load_csv("./data/sheets/6_train.csv")
        val_primitives, val_ground = load_csv("./data/sheets/6_val.csv")
        test_primitives, test_ground = load_csv("./data/sheets/6_test.csv")

        train_ground, val_ground, test_ground = convert_to_numeric_label(train_ground, val_ground, test_ground)

        return np.transpose(train_primitives), np.transpose(val_primitives), np.transpose(test_primitives), \
               train_ground, val_ground, test_ground




