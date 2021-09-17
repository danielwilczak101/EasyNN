from EasyNN.model import DenseNetwork, ReLU, SoftMax
import numpy as np

model = DenseNetwork(128, ReLU, 128, ReLU, 10, SoftMax)

x = np.random.uniform(-1, 1, 28 * 28)
y = model(x)

print(x)
print(y)
