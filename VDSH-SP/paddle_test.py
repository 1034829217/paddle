import paddle
from paddle.vision.models import mobilenet_v2

# build model
model = mobilenet_v2()

# build model and load imagenet pretrained weight
# model = mobilenet_v2(pretrained=True)

# build mobilenet v2 with scale=0.5
model = mobilenet_v2(scale=0.5)

x = paddle.rand([1, 3, 224, 224])
out = model(x)

print(out.shape)
