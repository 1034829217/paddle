import os
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix, csr_matrix, vstack

######################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Name of the dataset.")

args = parser.parse_args()
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

dataset = args.dataset
data_dir = '../dataset/{}'.format(dataset)
    
######################################################################################################################################

train_df = pd.read_pickle(os.path.join(data_dir, 'train.tf.df.pkl'))
test_df = pd.read_pickle(os.path.join(data_dir, 'test.tf.df.pkl'))
cv_df = pd.read_pickle(os.path.join(data_dir, 'cv.tf.df.pkl'))

print('num train: {} num test: {} num cv: {}'.format(len(train_df), len(test_df), len(cv_df)))

with open('../dataset/{}/vocab.pkl'.format(dataset), 'rb') as handle:
    vocabs = pickle.load(handle)
print('num vocabs: {}'.format(len(vocabs)))

print('num labels: {}'.format(len(set(list(train_df.label)))))