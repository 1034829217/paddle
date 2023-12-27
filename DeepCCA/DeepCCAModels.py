import paddle
import paddle.nn as nn
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Layer):
    def __init__(self, layer_sizes, input_size, name=None):
        super(MlpNet, self).__init__(name)
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            weight_attr = paddle.ParamAttr(
                name="%s_linear_%d_weight" % (self.full_name(), l_id),
                initializer=paddle.nn.initializer.Constant(value=0.5))
            bias_attr = paddle.ParamAttr(
                name="%s_linear_%d_bias" % (self.full_name(), l_id),
                initializer=paddle.nn.initializer.Constant(value=1.0))           
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1D(layer_sizes[l_id], momentum=0.1),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1], weight_attr=weight_attr, bias_attr=bias_attr),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1], weight_attr=weight_attr, bias_attr=bias_attr),
                    nn.Sigmoid(),
                    nn.BatchNorm1D(layer_sizes[l_id + 1], momentum=0.1),
                ))
        self.layers = nn.LayerList(layers)

    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Layer):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device='cpu'):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1, name="model1")
        self.model2 = MlpNet(layer_sizes2, input_size2, name="model2")

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x2)
        output2 = self.model2(x2)

        return output1, output2
