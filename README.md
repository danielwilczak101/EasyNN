![](https://raw.githubusercontent.com/danielwilczak101/EasyNN/media/images/readme_logo.png)

# EasyNN - Neural Networks made Easy
EasyNN is a python package designed to provide an easy-to-use Neural Network. The package is designed to work right out of the box, while also allowing the user to customize features as they see fit. 
### Requires Python version 3.9.7 or greater.

## Check out our [wiki](https://github.com/danielwilczak101/EasyNN/wiki) for more information.

## Installation:

Run python's pip3 to install:

```Python
pip3 install EasyNN
```

## Getting started with EasyNN(Basic Example):
To see more documention please see our wiki's infomation on the [number mnist](https://github.com/danielwilczak101/EasyNN/wiki/MNIST-Numbers) dataset.
```Python
from EasyNN.dataset.mnist.number import trained_model, dataset, show

# Downloads dataset to computer
train_data,train_labels,test_data,test_labels = dataset

# Grab a training data image
image = test_data[0]

# Uses the EasyNN train model on an example test image.
print(trained_model(image))

# Show the image
show(image, "image")
```

### Output:
```bash
Downloading Trained MNIST model...
Download complete.
Downloading number_train-images-idx3-ubyte.gz...
Downloading number_t10k-images-idx3-ubyte.gz...
Downloading number_train-labels-idx1-ubyte.gz...
Downloading number_t10k-labels-idx1-ubyte.gz...
Download complete.
Save complete.
7
```
### Image output
<p align="center">
  <img width="400px" height="400px" src="https://github.com/danielwilczak101/EasyNN/blob/media/images/number_7_example.png">
</p>

### Future goals for non known datasets:
```Python
from EasyNN.model import model

xtrain = "My images"
ytrain = "My labels"

model.dataset = xtrain, ytrain

model(xtrain[0])
```
