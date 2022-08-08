### 1. Setting up the model structure.

A model has to be first created. A dense network is simple enough to create. For classifying an input and getting probability outputs, `LogSoftMax` should be used for the last layer. In the following example, a network with an initial layer of 256 nodes, followed by the `ReLU` activation function, 256 more nodes, another `ReLU`, and finally 10 `LogSoftMax` outputs is given below.

```python
from EasyNN.model import LogSoftMax, Network, ReLU
model = Network(256, ReLU, 256, ReLU, 10, LogSoftMax)
```

### 2. Customizing attributes.

Next, attributes may be customized. Choosing the right attributes often requires some playing around with the model and seeing what ends up working best. For example, the starting learning rate may matter a lot.
```python
model.optimizer.lr = 0.01
```

### 3. Customizing callbacks.

An important customizability functionality are callbacks. These allow custom functions to be ran during training at specific steps.

One common callback is the `print` callback which allows progress to be seen during iterations. In the following example, every 20 iterations, the training accuracy is printed.
```python
model.print.on_training_start(iteration=True, accuracy=True, frequency=20)
```

### 4. Setting up the training data.

In order to train the model to some dataset, we need to set the dataset first. This can be done by simply setting the `training.data`.
```python
from EasyNN.examples.mnist.number.data import dataset

# Dataset format:
images = dataset[0]
labels = dataset[1]
# Saving the dataset:
model.training.data = dataset
```
**Note:** It is recommended that this is done in a separate file so that after the model is trained and saved, loading the model can be done without loading the dataset into memory.

### 5. Using the trained model.

Finally the model can be trained using `model.train()`, and it can be used using `model.classify(image)`.
```python
# Train the model.
model.train()
# Get an arbitrary image.
image = images[0]
# Classify the image.
print(model.classify(image))
```
Make sure the model is saved for future use, so that it does not need to be re-trained again.
```python
# Save the model.
model.save("mnist_numbers")

# Load and use a saved model.
from EasyNN import Model
model = Model.load("mnist_numbers")
```