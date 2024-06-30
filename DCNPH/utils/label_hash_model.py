import paddle
from paddle import nn
from paddle.nn import functional as F


class Label_net(nn.Layer):
    def __init__(self, label_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(Label_net, self).__init__()
        self.module_name = "text_model"
        cl1 = nn.Linear(label_dim, 4096)
        cl2 = nn.Linear(4096, bit)
        self.cl_text = nn.Sequential(
            cl1,
            nn.ReLU(),
            nn.BatchNorm1D(4096),
            cl2,
            nn.Tanh()
        )
    def forward(self, x):
        y = self.cl_text(x)
        return y


