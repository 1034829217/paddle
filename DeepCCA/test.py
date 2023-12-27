import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class MyModel(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        # Use paddle.nn.initializer.XavierNormal to initialize weights
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
        
        self.linear1 = nn.Linear(input_size, hidden_size, weight_attr=weight_attr)
        self.linear2 = nn.Linear(hidden_size, output_size, weight_attr=weight_attr)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

# Example usage
input_size = 10
hidden_size = 20
output_size = 5
device_ids = [ "gpu:6", "gpu:7"]
for device_id in device_ids:
        paddle.set_device(device_id)
model = MyModel(input_size, hidden_size, output_size)

# Forward pass to initialize the weights
dummy_input = paddle.randn([1, input_size], dtype='float32')
output = model(dummy_input)

# Print the weights
print("Linear 1 weights:")
print(model.linear1.weight.numpy())

print("\nLinear 2 weights:")
print(model.linear2.weight.numpy())

print(paddle.__version__)




