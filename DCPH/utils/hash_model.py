# import torch
# import torchvision
# from torchvision import models
import paddle
import paddle.nn as nn


from paddle.vision import models



LAYER1_NODE = 40960


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0.)

class HASH_Net(nn.Layer):
    def __init__(self, model_name, bit, pretrained=True):
        super(HASH_Net, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.features = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
        )
            
            # self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            cl3 = nn.Linear(4096, bit)

            print(dir(original_model._fc6), type(original_model._fc6))

            if pretrained:
                cl1.weight_attr = original_model._fc6.weight
                cl1.bias_attr = original_model._fc6.bias
                cl2.weight_attr = original_model._fc7.weight
                cl2.bias_attr = original_model._fc7.bias
            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(),
                nn.Dropout(),
                cl2,
                nn.ReLU(),
                cl3,
                nn.Tanh()
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained=True)
            self.features = original_model.features#ǰ���
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            cl3 = nn.Linear(4096, bit)

            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(),
                nn.Dropout(),
                cl2,
                nn.ReLU(),
                nn.Dropout(),
                cl3,
                nn.Tanh()
            )
            self.model_name = 'vgg11'

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = paddle.reshape(f, [f.shape[0], 256 * 6 * 6])
            f = self.classifier(f)
        else:
            f = paddle.reshape(f, [f.shape[0], -1])
            f = self.classifier(f)
        return f

