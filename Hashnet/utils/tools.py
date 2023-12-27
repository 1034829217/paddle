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

import numpy as np
import random
import paddle
from tqdm import tqdm

def set_random_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_result(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append(net(img).cpu())
    return paddle.concat(bs).sign(), paddle.concat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def save_database_code(model, database_loader, code_path, save=True):
    """save_database_code
    Compute bits code for database.
    Args:
        model
        database_loader
        code_path: the path where to save database_code.
        save: set as True by default.
    Returns: database_code
    """
    bs = []
    for img, _, _ in tqdm(database_loader, desc="Computing Database Code"):
        bs.append(model(img))
    database_code =  paddle.concat(bs).sign().numpy()
    if save:
        np.save(code_path, database_code)
        print("----- Save code and label of database in path: {}".format(code_path))
    return database_code
