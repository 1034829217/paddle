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

import paddle
import paddle.nn as nn

class HashNetLoss(nn.Layer):
    def __init__(self,
                 bit,
                 num_train,
                 alpha=0.1,
                 num_class=80):
        super(HashNetLoss, self).__init__()
        self.U = paddle.zeros([num_train, bit], dtype=paddle.float32)
        self.Y = paddle.zeros([num_train, num_class], dtype=paddle.float32)
        self.alpha = alpha
        self.scale = 1

    def forward(self, u, y, ind):
        u = paddle.tanh(self.scale * u)

        self.U[ind, :] = u
        self.Y[ind, :] = y.astype('float32')

        similarity = (y.astype('float32') @ self.Y.t() > 0).astype('float32')
        dot_product = self.alpha * u @ self.U.t()

        mask_positive = (similarity > 0).astype('float32')
        mask_negative = (similarity <= 0).astype('float32')

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clip(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.sum()
        S0 = mask_negative.sum()
        S = S0 + S1
        mask_positive = mask_positive.astype('bool')
        mask_negative = mask_negative.astype('bool')
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)
        loss = exp_loss.sum() / S
        return loss
