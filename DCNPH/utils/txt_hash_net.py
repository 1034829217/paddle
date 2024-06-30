# import torch
# from torch import nn
# from torch.nn import functional as F
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


import torch
from torch import nn
from torch.nn import functional as F
from utils.basic_module import BasicModule

LAYER1_NODE = 8192
LAYER2_NODE = 16384
LAYER3_NODE = 512

def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class TxtNet(BasicModule):
    def __init__(self, f_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtNet, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        self.conv1 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(f_dim, 1), stride=(1, 1))  # output:128*8192*1*1  [bs,oc,h,w]
        
        # self.conv2 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(f_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(LAYER1_NODE, LAYER1_NODE, kernel_size=(1,1), stride=(1, 1))
        # self.conv3 = nn.Conv2d(LAYER1_NODE, LAYER1_NODE, kernel_size=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))  # output:128*64*1*1
        self.apply(weights_init)

    def forward(self, y1):
        #print("#shapey1input:",y1.shape)#shapey1input: torch.Size([4, 1, 2000, 1])
        y1 = self.conv1(y1)  # x:128*1*1386
        #print("#shapey1:",y1.shape) #shapey1: torch.Size([4, 4096, 1, 1])
        y1 = F.relu(y1)
        #y2 = self.conv2(y2)  # x:128*1*1386
        #y2 = F.relu(y2)
        #y = torch.cat([y1,y2],1).cuda()
        #y = self.conv3(y)
        #y = y.squeeze()  # ����ά��
        #y=torch.tanh(y)
        y1 = self.conv2(y1)
        y1 = F.relu(y1)
        # y1 = self.conv3(y1)
        # y1 = F.relu(y1)
        #print("#shapey12:",y1.shape)
        y1 = self.conv4(y1)
        y1 = y1.squeeze()  # ����ά��
        y1=torch.tanh(y1)
        #print("#shapey1f:",y1.shape)
        return y1  # 128*64

class Txt_net(BasicModule):
    def __init__(self,y_dim,bit):
        super(Txt_net, self).__init__()
        self.y_dim=y_dim
        self.interp_block1=nn.Sequential(
            nn.AvgPool2d(kernel_size=(50,1), stride=(50,1)),
            nn.Conv2d(1,1,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(inplace=True)

        )
        self.interp_block2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(30,1), stride=(30,1)),
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)

        )
        self.interp_block3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(15,1), stride=(15,1)),
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)

        )
        self.interp_block4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(10,1), stride=(10,1)),
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)

        )
        self.interp_block5 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5,1), stride=(5,1)),
            nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)

        )
        self.feature =nn.Sequential(
            nn.Conv2d(6,4096,kernel_size=(y_dim,1),stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4096),
            nn.Conv2d(4096, 512, kernel_size=(1, 1), stride=(1, 1)),#eg[in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1]
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, bit, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, y):
        y1=self.upsample(self.interp_block1(y))
        y2=self.upsample(self.interp_block2(y))
        y3=self.upsample(self.interp_block3(y))
        y4=self.upsample(self.interp_block4(y))
        y5=self.upsample(self.interp_block5(y))
        y1=y1.cuda()
        y2=y2.cuda()
        y3=y3.cuda()
        y4=y4.cuda()
        y5=y5.cuda()
        y=torch.cat([y,y1,y2,y3,y4,y5],1).cuda()
        y=self.feature(y)
        y=y.squeeze()
        return y

    def upsample(self,y):
        output=F.interpolate(input=y,size=(self.y_dim,1),mode='bilinear',align_corners=True)
        return output






