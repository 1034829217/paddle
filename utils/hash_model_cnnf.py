import torch
import time
import scipy.io as scio
from torch import nn
import torch.nn.functional as F


class BasicModule(torch.nn.Module):
    """
    封装nn.Module，主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), 'MyCheckpoint/' + name)
        path = 'Mycheckpoint/' + name
        return path

    def forward(self, *input):
        pass

class ImageNet(BasicModule):
    def __init__(self, bit, keep_prob = 0.5):
        super(ImageNet, self).__init__()
        self.keep_prob = keep_prob
        self.data = scio.loadmat('/data/trc/imagenet-vgg-f.mat')
        self.weights = self.data['layers'][0]
        self.module_name = 'image_model'
        self.feature = 512
        self.bit = bit
        self.n_fc7 = 4096
        # self.mean = torch.from_numpy(self.data['meta'][0][0][2][0][0][2].transpose()).type(torch.float)
        conv1_w, conv1_b = self.weights[0][0][0][2][0]
        conv1_b = conv1_b.reshape(-1)
        self.conv1_w = torch.nn.Parameter(torch.FloatTensor(conv1_w).permute(3, 2, 0, 1))
        self.conv1_b = torch.nn.Parameter(torch.FloatTensor(conv1_b))
        self.pad1 = self.weights[0][0][0][4][0]
        self.stride1 = self.weights[0][0][0][5][0]

        # pool1参数
        self.stride_pool1 = self.weights[3][0][0][4][0]
        self.pad_pool1 = self.weights[3][0][0][5][0]
        self.area_pool1 = self.weights[3][0][0][3][0]

        # conv2参数
        conv2_w, conv2_b = self.weights[4][0][0][2][0]
        conv2_b = conv2_b.reshape(-1)
        self.conv2_w = torch.nn.Parameter(torch.FloatTensor(conv2_w).permute(3, 2, 0, 1))
        self.conv2_b = torch.nn.Parameter(torch.FloatTensor(conv2_b))
        self.pad2 = self.weights[4][0][0][4][0]
        self.stride2 = self.weights[4][0][0][5][0]

        # pool2参数
        self.stride_pool2 = self.weights[7][0][0][4][0]
        self.pad_pool2 = self.weights[7][0][0][5][0]
        self.area_pool2 = self.weights[7][0][0][3][0]

        # conv3参数
        conv3_w, conv3_b = self.weights[8][0][0][2][0]
        conv3_b = conv3_b.reshape(-1)
        self.conv3_w = torch.nn.Parameter(torch.FloatTensor(conv3_w).permute(3, 2, 0, 1))
        self.conv3_b = torch.nn.Parameter(torch.FloatTensor(conv3_b))
        self.pad3 = self.weights[8][0][0][4][0]
        self.stride3 = self.weights[8][0][0][5][0]

        # conv4参数
        conv4_w, conv4_b = self.weights[10][0][0][2][0]
        conv4_b = conv4_b.reshape(-1)
        self.conv4_w = torch.nn.Parameter(torch.FloatTensor(conv4_w).permute(3, 2, 0, 1))
        self.conv4_b = torch.nn.Parameter(torch.FloatTensor(conv4_b))
        self.pad4 = self.weights[10][0][0][4][0]
        self.stride4 = self.weights[10][0][0][5][0]

        # conv5参数
        conv5_w, conv5_b = self.weights[12][0][0][2][0]
        conv5_b = conv5_b.reshape(-1)
        self.conv5_w = torch.nn.Parameter(torch.FloatTensor(conv5_w).permute(3, 2, 0, 1))
        self.conv5_b = torch.nn.Parameter(torch.FloatTensor(conv5_b))
        self.pad5 = self.weights[12][0][0][4][0]
        self.stride5 = self.weights[12][0][0][5][0]

        # pool5参数
        self.stride_pool5 = self.weights[14][0][0][4][0]
        self.pad_pool5 = self.weights[14][0][0][5][0]
        self.area_pool5 = self.weights[14][0][0][3][0]

        # fc6参数
        fc6_w, fc6_b = self.weights[15][0][0][2][0]
        fc6_b = fc6_b.reshape(-1)
        self.fc6_w = torch.nn.Parameter(torch.FloatTensor(fc6_w).permute(3, 2, 0, 1))
        self.fc6_b = torch.nn.Parameter(torch.FloatTensor(fc6_b))

        # fc7参数
        fc7_w, fc7_b = self.weights[17][0][0][2][0]
        fc7_b = fc7_b.reshape(-1)
        self.fc7_w = torch.nn.Parameter(torch.FloatTensor(fc7_w).permute(3, 2, 0, 1))
        self.fc7_b = torch.nn.Parameter(torch.FloatTensor(fc7_b))

        # fc8 image_feature
        # self.fc8 = torch.nn.Conv2d(in_channels=self.n_fc7, out_channels=self.feature, kernel_size=1)

        # fc9 image_hash
        self.hashlayer = torch.nn.Conv2d(in_channels=self.n_fc7, out_channels=self.bit, kernel_size=1)

    def forward(self, x):
        # if x.is_cuda:
        #     x = x - self.mean.cuda()
        # else:
        #     x = x - self.mean

        dim = [int(self.pad1[2]), int(self.pad1[3]), int(self.pad1[0]), int(self.pad1[1]), 0, 0, 0, 0]
        x = F.pad(x, dim, 'constant')

        x = F.conv2d(x, self.conv1_w, self.conv1_b, stride=int(self.stride1[0]))
        x = F.relu(x)

        x = F.local_response_norm(x, 2, alpha=0.0001, beta=0.75, k=2.000)

        # 池化1
        # pytorch中kernel_size为max_pool窗口大小, stride为滑动窗口大小
        dim = [int(self.pad_pool1[2]), int(self.pad_pool1[3]), int(self.pad_pool1[0]), int(self.pad_pool1[1]), 0, 0, 0, 0]
        x = F.pad(x, dim, 'constant')
        x = F.max_pool2d(x, kernel_size=(int(self.area_pool1[0]), int(self.area_pool1[1])),
                         stride=(int(self.stride_pool1[0]), int(self.stride_pool1[1])))

        dim = [int(self.pad2[2]), int(self.pad2[3]), int(self.pad2[0]), int(self.pad2[1]), 0, 0, 0, 0]
        x = F.pad(x, dim, 'constant')
        x = F.conv2d(x, self.conv2_w, self.conv2_b, stride=int(self.stride2[0]))
        x = F.relu(x)
        x = F.local_response_norm(x, 2, alpha=0.0001, beta=0.75, k=2.000)
        dim = [int(self.pad_pool2[2]), int(self.pad_pool2[3]), int(self.pad_pool2[0]), int(self.pad_pool2[1]), 0, 0, 0, 0]
        x = F.pad(x, dim, 'constant')
        x = F.max_pool2d(x, kernel_size=(int(self.area_pool2[0]), int(self.area_pool2[1])),
                         stride=(int(self.stride_pool2[0]), int(self.stride_pool2[1])))

        dim = [int(self.pad3[2]), int(self.pad3[3]), int(self.pad3[0]), int(self.pad3[1]), 0, 0, 0, 0]
        x = F.pad(x, dim, 'constant')
        x = F.conv2d(x, self.conv3_w, self.conv3_b, stride=int(self.stride3[0]))
        x = F.relu(x)

        dim = [int(self.pad4[2]), int(self.pad4[3]), int(self.pad4[0]), int(self.pad4[1]), 0, 0, 0, 0]
        x = F.pad(x, dim, 'constant')
        x = F.conv2d(x, self.conv4_w, self.conv4_b, stride=int(self.stride4[0]))
        x = F.relu(x)

        dim = [int(self.pad5[2]), int(self.pad5[3]), int(self.pad5[0]), int(self.pad5[1]), 0, 0, 0, 0]
        x = F.pad(x, dim, 'constant')
        x = F.conv2d(x, self.conv5_w, self.conv5_b, stride=int(self.stride5[0]))
        x = F.relu(x)
        dim = [int(self.pad_pool5[2]), int(self.pad_pool5[3]), int(self.pad_pool5[0]), int(self.pad_pool5[1]), 0, 0, 0, 0]
        x = F.pad(x, dim, 'constant')
        x = F.max_pool2d(x, kernel_size=(int(self.area_pool5[0]), int(self.area_pool5[1])),
                         stride=(int(self.stride_pool5[0]), int(self.stride_pool5[1])))

        # fc6和fc7
        x = F.conv2d(x, self.fc6_w, self.fc6_b, stride=1)
        x = F.relu(x)
        x = F.dropout(x, p=self.keep_prob)
        x = F.conv2d(x, self.fc7_w, self.fc7_b, stride=1)
        x = F.relu(x)
        x = F.dropout(x, p=self.keep_prob)

        # image_feature
        # x = F.relu(self.fc8(x))

        # image hash code
        hashcode = self.hashlayer(x)
        hashcode = F.tanh(hashcode)
        return hashcode.squeeze()
