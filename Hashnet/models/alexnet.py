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

"""
This script is based on:
https://github.com/littletomatodonkey/AlexNet-Prod/blob/master/pipeline/Step5/AlexNet_paddle/paddlevision/models/alexnet.py
And it's worth noting that it will automatically transfer the model params from PyTorch style.
"""
import os
import math
import paddle
import paddle.nn as nn
from typing import Any
from paddle import ParamAttr
from paddle.nn.initializer import Uniform

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Layer):
    def __init__(self, num_classes: int=1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2D(
                3,
                64,
                kernel_size=11,
                stride=4,
                padding=2,
                weight_attr=ParamAttr(initializer=self.uniform_init(3 * 11 *
                                                                    11))),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2),
            nn.Conv2D(
                64,
                192,
                kernel_size=5,
                padding=2,
                weight_attr=ParamAttr(initializer=self.uniform_init(64 * 5 *
                                                                    5))),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2),
            nn.Conv2D(
                192,
                384,
                kernel_size=3,
                padding=1,
                weight_attr=ParamAttr(initializer=self.uniform_init(192 * 3 *
                                                                    4))),
            nn.ReLU(),
            nn.Conv2D(
                384,
                256,
                kernel_size=3,
                padding=1,
                weight_attr=ParamAttr(initializer=self.uniform_init(384 * 3 *
                                                                    3))),
            nn.ReLU(),
            nn.Conv2D(
                256,
                256,
                kernel_size=3,
                padding=1,
                weight_attr=ParamAttr(initializer=self.uniform_init(256 * 3 *
                                                                    3))),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2), )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                256 * 6 * 6,
                4096,
                weight_attr=ParamAttr(initializer=self.uniform_init(256 * 6 *
                                                                    6))),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(
                4096,
                4096,
                weight_attr=ParamAttr(initializer=self.uniform_init(4096))),
            nn.ReLU(),
            nn.Linear(
                4096,
                num_classes,
                weight_attr=ParamAttr(initializer=self.uniform_init(4096))), )

    def uniform_init(self, num):
        stdv = 1.0 / math.sqrt(num)
        return Uniform(-stdv, stdv)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x


def transfer(output_fp):
    model_urls_torch = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }
    from torch.hub import load_state_dict_from_url
    torch_dict = load_state_dict_from_url(model_urls_torch['alexnet'],
                                              progress=True)
    paddle_dict = {}
    fc_names = [
        "classifier.1.weight", "classifier.4.weight", "classifier.6.weight"
    ]
    for key in torch_dict:
        weight = torch_dict[key].cpu().detach().numpy()
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight {} need to be trans".format(key))
            weight = weight.transpose()
        paddle_dict[key] = weight
    paddle.save(paddle_dict, output_fp)

def load_dygraph_pretrain(model, path=None):
    """
    model:network
    path: str or bool, default=None.
    """
    if isinstance(path, bool) and path:
        path = os.path.join(os.path.abspath(r"."), "models/AlexNet_pretrained.pdparams")
    if not os.path.exists(path):
        print("Model pretrain path {} does not "
                        "exists.".format(path))
        print("Transferring state_dict from torch ...")
        transfer(path)
    param_state_dict = paddle.load(path)
    model.set_dict(param_state_dict)
    print("Loading AlexNet state from path: {}".format(path))
    return


def alexnet(pretrained: bool=False, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (str): Pre-trained parameters of the model on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        load_dygraph_pretrain(model, pretrained)
    return model