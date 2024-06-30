import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import pickle



# class DatasetProcessingNUS_WIDE(Dataset):
#     def __init__(self, t_len, data_path, img_filename, sentence_vector, word2vec_file, label_filename, ind_filename, transform=None):
#         self.img_path = data_path
#         self.transform = transform
#         img_filepath = os.path.join(data_path, img_filename)
#         fp = open(img_filepath, 'r')
#         self.img_filename = [x.strip() for x in fp]
#         fp.close()
#         ind_filepath = os.path.join(data_path, ind_filename)
#         fp = open(ind_filepath, 'r')
#         self.ind_list = [x.strip() for x in fp]
#         fp.close()
#         vector_filepath = os.path.join(data_path, sentence_vector)
#         fp = open(vector_filepath, 'r')
#         self.vector_list = [x.strip().split() for x in fp]
#         fp.close()
#         word2vec_path = os.path.join(data_path, word2vec_file)
#         with open(word2vec_path, 'rb') as f:
#             self.word2vec = pickle.load(f)
#         label_filepath = os.path.join(data_path, label_filename)
#         self.label = np.loadtxt(label_filepath, dtype=np.int64)
#         self.max_len = t_len
#
#     def __getitem__(self, index):
#         ind = int(self.ind_list[index]) - 1
#         vector_str = self.vector_list[ind]
#         y_matrix = np.zeros((self.max_len, 300))
#         for i in range(len(vector_str)):
#             v_id = int(vector_str[i])
#             y_matrix[i, :] = self.word2vec[v_id]
#         for i in range(len(vector_str), self.max_len):
#             y_matrix[i, :] = self.word2vec[0]
#         img = Image.open(os.path.join(self.img_path, self.img_filename[ind]))
#         img = img.convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         label = torch.from_numpy(self.label[ind])
#         y_matrix = torch.from_numpy(y_matrix).type(torch.FloatTensor)
#         return img, y_matrix, label, index
#
#     def __len__(self):
#         return len(self.ind_list)
class DatasetProcessingNUS_WIDE_label(Dataset):
    def __init__(self, data_path, label_filename):
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath)
        self.ind_list = self.label.shape[0]

    def __getitem__(self, index):
        label = torch.from_numpy(self.label[index])
        return label, index

    def __len__(self):
        return self.ind_list

class DatasetProcessingNUS_WIDE(Dataset):
    # def __init__(self, data_path, img_filename, txt_filename, label_filename, ind_filename, transform=None):
    def __init__(self, data_path, img_filename, txt_filename, feature_filename, label_filename, ind_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        ind_filepath = os.path.join(data_path, ind_filename)
        fp = open(ind_filepath, 'r')
        self.ind_list = [x.strip() for x in fp]
        fp.close()
        vector_filepath = os.path.join(data_path, txt_filename)
        fp = open(vector_filepath, 'r')
        self.vector = np.asarray([[float(i) for i in val.strip().split(' ')] for val in fp.readlines()])
        fp.close()
        feature_filepath = os.path.join(data_path, feature_filename)
        fp = open(feature_filepath, 'r')
        self.feature = np.asarray([[float(i) for i in val.strip().split(' ')] for val in fp.readlines()])
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        fp = open(label_filepath, 'r')
        self.label = np.asarray([[int(i) for i in val.strip().split(' ')] for val in fp.readlines()])
        fp.close()

    def __getitem__(self, index):
        ind = int(self.ind_list[index]) - 1
        img = Image.open(os.path.join(self.img_path, self.img_filename[ind]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[ind])
        
        # vector = self.vector[ind, :]
        # print("vector",self.vector.shape,"ind:",ind)
        
        vector = torch.Tensor(self.vector[ind][np.newaxis, :, np.newaxis])
        feature = torch.Tensor(self.feature[ind][np.newaxis, :, np.newaxis])
        return img, vector, feature, label, index
        # return img, vector, label, index

    def __len__(self):
        return len(self.ind_list)

class DatasetProcessingNUS_WIDE_txt(Dataset):
    def __init__(self, data_path, txt_vector, ind_filename, label_filename):
        self.txt_path = data_path
        ind_filepath = os.path.join(data_path, ind_filename)
        fp = open(ind_filepath, 'r')
        self.ind_list = [x.strip() for x in fp]
        fp.close()
        vector_filepath = os.path.join(data_path, txt_vector)
        self.vector = np.loadtxt(vector_filepath, dtype=np.float)
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        ind = int(self.ind_list[index]) - 1
        vector = self.vector[ind, :]
        label = torch.from_numpy(self.label[ind])
        return vector, label, index

    def __len__(self):
        return len(self.ind_list)

class DatasetProcessingMS_COCO(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)

