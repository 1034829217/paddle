import os
from os.path import join
import numpy as np
import torch
#from scipy.sparse import csr_matrix
import pandas as pd
import pickle
from torch.utils.data import Dataset

##########################################################################################################################

class SingleLabelTextDataset(Dataset):
    """datasets wrapper for ng20, agnews, dbpedia, yahooanswer"""

    def __init__(self, data_dir, download=False, subset='train', bow_format='tf'):
        """
        Args:
            data_dir (string): Directory for loading and saving train, test, and cv dataframes.
            download (boolean): Download newsgroups20 dataset from sklearn if necessary.
            subset (string): Specify subset of the datasets. The choices are: train, test, cv.
            bow_format (string): A weight scheme of a bag-of-words document. The choices are:
                tf (term frequency), tfidf (term freq with inverse document frequency), bm25.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.bow_format = bow_format
        self.df = self.load_df('{}.{}.df.pkl'.format(subset, bow_format))

    def load_df(self, df_file):
        df_file = os.path.join(self.data_dir, df_file)
        return pd.read_pickle(df_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        doc_bow = self.df.iloc[idx].bow
        doc_bow = torch.from_numpy(doc_bow.toarray().squeeze().astype(np.float32))
        label = self.df.iloc[idx].label
        return (doc_bow, label)
    
    def num_classes(self):
        return len(set(self.df.label))
    
    def num_features(self):
        return self.df.bow.iloc[0].shape[1]
    
##########################################################################################################################

class MultiLabelTextDataset(SingleLabelTextDataset):
    """datasets wrapper for reuters, rcv1, tmc"""
    def __init__(self, data_dir, download=False, subset='train', bow_format='tf'):
        super(MultiLabelTextDataset, self).__init__(data_dir, download, subset, bow_format)
        
    def __getitem__(self, idx):
        doc_bow = self.df.iloc[idx].bow
        doc_bow = torch.from_numpy(doc_bow.toarray().squeeze().astype(np.float32))
        label_bow = self.df.iloc[idx].label
        label_bow = torch.from_numpy(label_bow.toarray().squeeze().astype(np.float32))
        return (doc_bow, label_bow)
    
    def num_classes(self):
        return self.df.iloc[0].label.shape[1]