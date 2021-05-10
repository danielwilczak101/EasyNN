![](https://raw.githubusercontent.com/danielwilczak101/EasyNN/media/images/readme_logo.png)

# EasyNN - Neural Networks made Easy

This project is still in work and does not have a production version yet.

## Number MNIST Example user code:

#### Pre-trained:
```Python
from EasyNN.examples.mnist import images, labels, show, model

# Shows me the first image in the dataset
show(images[0])

# Tells me what the image is.
model(images[0])
```

#### Not trained:
```Python
from EasyNN.examples.mnist import images, labels, show
from EasyNN import Classifier

# Train several models and take the best one.
model = Classifier(images, labels)

# Shows me the first image in the dataset.
show(images[0])

# Tells me what the image is.
model(images[0])
```

## CIFAR Example user code:
```Python
#In work currently
```
