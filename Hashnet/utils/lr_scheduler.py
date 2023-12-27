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

from paddle.optimizer.lr import LRScheduler


class DecreaseLRScheduler(LRScheduler):
    def __init__(self,
                 learning_rate,
                 start_lr,
                 de_step=50,
                 gamma=0.5,
                 last_epoch=-1):
        """init DecreaseLRScheduler """
        self.start_lr = start_lr
        self.de_step = de_step
        self.gamma = gamma
        super(DecreaseLRScheduler, self).__init__(learning_rate, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        val = self.start_lr * (self.gamma ** (t // self.de_step))
        return val
