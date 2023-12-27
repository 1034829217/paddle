#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image
from paddle.vision import transforms
from paddle.vision import datasets as dsets
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

def config_dataset(config):
    """
    get config.data: list_path
    """
    if "cifar" in config.dataset:
        config.topK = -1
        config.num_class = 10
    elif config.dataset in ["nuswide_21", "nuswide_21_m"]:
        config.topK = 5000
        config.num_class = 21
    elif config.dataset == "nuswide_81_m":
        config.topK = 5000
        config.num_class = 81
    elif config.dataset == "coco":
        config.topK = 5000
        config.num_class = 80
    elif config.dataset == "coco_lite":
        config.topK = 5000
        config.num_class = 80
    elif config.dataset == "imagenet":
        config.topK = 1000
        config.num_class = 100
    elif config.dataset == "mirflickr":
        config.topK = -1
        config.num_class = 38
    elif config.dataset == "voc2012":
        config.topK = -1
        config.num_class = 20

    config.data = {
        "train_set": {"list_path": os.path.join("data", config.dataset, "train.txt"), "batch_size": config.batch_size},
        "database": {"list_path": os.path.join("data", config.dataset, "database.txt"), "batch_size": config.batch_size},
        "test": {"list_path": os.path.join("data", config.dataset, "test.txt"), "batch_size": config.batch_size}}
    return config

class ImageList(Dataset):
    def __init__(self, data_path, image_list, transform):
        self.imgs = [(os.path.join(data_path, val.split()[0]), np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip()]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.Cifar10):
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))
        if self.transform is not None:
            image = self.transform(image)

        label = np.eye(10, dtype=np.float32)[np.array(label)]
        if self.backend == 'pil':
            return image, label, idx

        return image.astype(self.dtype), label, idx


def get_dataloader(config, dataset, mode='train', multi_process=False, drop_last=False):
    """Get dataloader with config, dataset, mode as input, allows multiGPU settings.

        Multi-GPU loader is implements as distributedBatchSampler.

    Args:
        config: see config.py for details
        dataset: paddle.io.dataset object
        mode: train/val
        multi_process: if True, use DistributedBatchSampler to support multi-processing
    Returns:
        dataloader: paddle.io.DataLoader object.
    """

    batch_size = config.batch_size

    if multi_process is True:
        sampler = DistributedBatchSampler(dataset,
                                          batch_size=batch_size,
                                          shuffle=(mode == 'train'),
                                          drop_last=drop_last,
                                          )
        dataloader = DataLoader(dataset,
                                batch_sampler=sampler,
                                num_workers=4)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                shuffle=(mode == 'train'),
                                drop_last=drop_last)
    return dataloader

def cifar_dataset(config):
    train_size = 500
    test_size = 100

    if config.dataset == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = MyCIFAR10(mode='train',
                              transform=transform)

    test_dataset = MyCIFAR10(mode='test',
                             transform=transform)

    database_dataset = MyCIFAR10(mode='test',
                                 transform=transform)

    X = np.concatenate((np.array(train_dataset.data)[:, 0], np.array(test_dataset.data)[:, 0]))
    L = np.concatenate((np.array(train_dataset.data)[:, 1], np.array(test_dataset.data)[:, 1]))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config.dataset == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config.dataset == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config.dataset == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset_image = X[train_index]
    train_dataset_label = L[train_index]
    train_dataset.data = []
    for i in range(len(train_dataset_image)):
        train_dataset.data.append((train_dataset_image[i], train_dataset_label[i]))

    test_dataset_image = X[test_index]
    test_dataset_label = L[test_index]
    test_dataset.data = []
    for i in range(len(test_dataset_image)):
        test_dataset.data.append((test_dataset_image[i], test_dataset_label[i]))

    database_dataset_image = X[database_index]
    database_dataset_label = L[database_index]
    database_dataset.data = []
    for i in range(len(database_dataset_image)):
        database_dataset.data.append((database_dataset_image[i], database_dataset_label[i]))

    return train_dataset, test_dataset, database_dataset


def get_data(config):
    if "cifar" in config.dataset:
        return cifar_dataset(config)

    dsets = {}
    # dset_loaders = {}
    data_config = config.data

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config.data_path,
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config.resize_size, config.crop_size, data_set))
        print(data_set, len(dsets[data_set]))
    return dsets["train_set"], dsets["test"], dsets["database"]
