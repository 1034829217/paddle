import paddle
import paddle.nn as nn
import paddle.nn.initializer as init


class DSH(nn.Layer):
    def __init__(self, num_binary):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=5, padding=2),  # same padding
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),

            nn.Conv2D(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=3, stride=2),

            nn.Conv2D(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 500),
            nn.ReLU(),

            nn.Linear(500, num_binary)
        )

        for m in self.named_sublayers():
            if m.__class__ == nn.Conv2D or m.__class__ == nn.Linear:
                init.XavierNormal(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        # print(type(x), type(x.size), type(x.shape))
        # print(type(x), x.shape[0])
        x = paddle.reshape(x, shape=[x.shape[0],-1])
        x = self.fc(x)

        return x
