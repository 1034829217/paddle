# import paddle
# from paddle import nn
# from paddle.nn import functional as F
#
# LAYER1_NODE = 40960
#
# class TxtNet(nn.Module):
#     def __init__(self, y_dim, bit):
#         """
#         :param y_dim: dimension of tags
#         :param bit: bit number of the final binary code
#         """
#         super(TxtNet, self).__init__()
#         self.module_name = "text_model"
#         cl1 = nn.Linear(y_dim, 2048)
#         cl2 = nn.Linear(2048, bit)
#         self.cl_text = nn.Sequential(
#             cl1,
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.BatchNorm1d(2048),
#             cl2,
#             nn.Tanh(),
#         )
#     def forward(self, x):
#         y = self.cl_text(x)
#         return y


import paddle
from paddle import nn
from paddle.nn import functional as F
from utils.basic_module import BasicModule

LAYER1_NODE = 8192
LAYER2_NODE = 16384

# def weights_init(m):
#     if type(m) == nn.Conv2d:
#         nn.init.normal_(m.weight.data, 0.0, 0.01)
#         nn.init.normal_(m.bias.data, 0.0, 0.01)

def weights_init(m):
    if isinstance(m, nn.Conv2D):
        # Initialize weights using normal distribution with mean 0 and standard deviation 0.01
        initializer = nn.initializer.Normal(mean=0.0, std=0.01)
        initializer(m.weight)
        # Initialize bias using normal distribution with mean 0 and standard deviation 0.01
        initializer(m.bias)


class TxtNet(BasicModule):
    def __init__(self, f_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtNet, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        self.conv1 = nn.Conv2D(1, LAYER1_NODE, kernel_size=(f_dim, 1), stride=(1, 1))  # output:128*8192*1*1  [bs,oc,h,w]
        #self.conv2 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(f_dim, 1), stride=(1, 1))
        self.conv3 = nn.Conv2D(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))  # output:128*64*1*1
        self.apply(weights_init)

    def forward(self, y1):
        y1 = self.conv1(y1)  # x:128*1*1386
        y1 = F.relu(y1)
        #y2 = self.conv2(y2)  # x:128*1*1386
        #y2 = F.relu(y2)
        #y = paddle.cat([y1,y2],1).cuda()
        #y = self.conv3(y)
        #y = y.squeeze()  # 减少维度
        #y=paddle.tanh(y)
        
        y1 = self.conv3(y1)
        y1 = y1.squeeze()  # 减少维度
        y1=paddle.tanh(y1)
        return y1  # 128*64

class Txt_net(BasicModule):
    def __init__(self,y_dim,bit):
        super(Txt_net, self).__init__()
        self.y_dim=y_dim
        self.interp_block1=nn.Sequential(
            nn.AvgPool2D(kernel_size=(50,1), stride=(50,1)),
            nn.Conv2D(1,1,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU()

        )
        self.interp_block2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=(30,1), stride=(30,1)),
            nn.Conv2D(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()

        )
        self.interp_block3 = nn.Sequential(
            nn.AvgPool2D(kernel_size=(15,1), stride=(15,1)),
            nn.Conv2D(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()

        )
        self.interp_block4 = nn.Sequential(
            nn.AvgPool2D(kernel_size=(10,1), stride=(10,1)),
            nn.Conv2D(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()

        )
        self.interp_block5 = nn.Sequential(
            nn.AvgPool2D(kernel_size=(5,1), stride=(5,1)),
            nn.Conv2D(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)

        )
        self.feature =nn.Sequential(
            nn.Conv2D(6,4096,kernel_size=(y_dim,1),stride=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2D(4096),
            nn.Conv2D(4096, 512, kernel_size=(1, 1), stride=(1, 1)),#eg[in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1]
            nn.ReLU(),
            nn.BatchNorm2D(512),
            nn.Conv2D(512, bit, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, y):
        y1=self.upsample(self.interp_block1(y))
        y2=self.upsample(self.interp_block2(y))
        y3=self.upsample(self.interp_block3(y))
        y4=self.upsample(self.interp_block4(y))
        y5=self.upsample(self.interp_block5(y))
        y1 = paddle.to_tensor(y1)
        y2 = paddle.to_tensor(y2)
        y3 = paddle.to_tensor(y3)
        y4 = paddle.to_tensor(y4)
        y5 = paddle.to_tensor(y5)
        y=paddle.concat([y, y1, y2, y3, y4, y5], axis=1)
        y=self.feature(y)
        y=paddle.squeeze(y)
        return y

    def upsample(self,y):
        output=F.interpolate(input=y,size=(self.y_dim,1),mode='bilinear',align_corners=True)
        return output






