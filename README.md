![](https://raw.githubusercontent.com/danielwilczak101/EasyNN/media/images/readme_logo.png)

# EasyNN - Neural Networks made Easy
EasyNN is a python package designed to provide an easy-to-use Neural Network. The package is designed to work right out of the box with multiple datasets, while also allowing the user to customize features as they see fit. 
### EasyNN requires Python version 3.9.7 or greater.

## See our [wiki](https://github.com/danielwilczak101/EasyNN/wiki) for more information.

## Installation:

Run python's pip3 to install:

```Python
pip3 install EasyNN
```

## Minimum Required
```Python
from EasyNN.examples.mnist.fashion.trained import model

# Classify the an image in the dataset
print(model.classify(image))
```

## Getting started with EasyNN(Basic Example):
To see more documention please see our wiki's infomation on the [number mnist](https://github.com/danielwilczak101/EasyNN/wiki/MNIST-Numbers) dataset.
```Python
from EasyNN.examples.mnist.number.trained import model
from EasyNN.utilities.image.preprocess import image
from EasyNN.utilities.download import download

# Download an example image.
download("four.jpg","https://bit.ly/3lAJrMe")

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

# Converting your image into the correct format for the mnist number dataset.
image = image("four.jpg").format(**format_options)

print(model.classify(image))

model.show(image)
```

### Output:
```bash
Downloading - number_parameters.npz:
[################################] 1769/1769 - 00:00:00
Downloading - number_structure.pkl:
[################################] 10700/10700 - 00:00:00
Downloading - four.jpg:
[################################] 1371/1371 - 00:00:00
4
```
### Image output:
<p align="center">
  <img width="400px" height="400px" src="https://github.com/danielwilczak101/EasyNN/blob/media/images/example_four.png">
</p>

## To see more examples with many other datasets. Please visit our [wiki](https://github.com/danielwilczak101/EasyNN/wiki).
