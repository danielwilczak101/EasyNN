![](https://raw.githubusercontent.com/danielwilczak101/EasyNN/media/images/readme_logo.png)

# EasyNN - Neural Networks made Easy
EasyNN is a python package designed to provide an easy-to-use Neural Network. The package is designed to work right out of the box with multiple datasets, while also allowing the user to customize features as they see fit. 
### EasyNN requires Python version 3.9.7 or greater.

## See our [wiki](https://github.com/danielwilczak101/EasyNN/wiki) for more information and [Datasets](https://github.com/danielwilczak101/EasyNN/wiki).

## Installation:

Run python's pip3 to install:

```Python
pip3 install EasyNN
```

### Model:
```Python
from EasyNN.examples.mnist.number.trained import model

# Classify an image.
print(model.classify(image))
```

### Dataset Example:
```Python
from EasyNN.examples.mnist.number.trained import model
from EasyNN.examples.mnist.number.data import dataset

images, labels = dataset

# Classify what the second image is in the dataset.
print(model.classify(images[1]))

# Show the image.
model.show(images[1])
```

### Dataset example output:
```
Downloading - number_parameters.npz:
[################################] 1769/1769 - 00:00:00
Downloading - number_structure.pkl:
[################################] 10700/10700 - 00:00:00
Downloading - number_dataset.npz:
[################################] 11221/11221 - 00:00:00
0
```

### Full example:
More info can be found about [converting images](https://github.com/danielwilczak101/EasyNN/wiki/Image-Utility) in the utilities section.
```Python
from EasyNN.examples.mnist.number.trained import model
from EasyNN.utilities import Preprocess, download

# Download an example image.
download("three.jpg","https://bit.ly/3dbO1eV")

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=30,
    resize=(28, 28),
    rotate=3,
)

# Converting your image into the correct format for the mnist number dataset.
image = Preprocess("three.jpg").format(**format_options)

# Classify what the image is using the pretrained model.
print(model.classify(image))

# Show the image after it has been processed.
model.show(image)
```
### Output:
```bash
Downloading - four.jpg:
[################################] 1371/1371 - 00:00:00
3
```

### Image output:
<p align="center">
  <img width="400px" height="400px" src="https://raw.githubusercontent.com/danielwilczak101/EasyNN/media/images/example_three.png">
</p>

### Trained Models
Use the trained models section to see EasyNN's datasets and pre-trained neural networks ready to run.  
<br />
[MNIST Number](https://github.com/danielwilczak101/EasyNN/wiki/MNIST-Numbers) Classifier network for images of handwritten single digits between 0 and 9.  
[MNIST Fashion](https://github.com/danielwilczak101/EasyNN/wiki/MNIST-Fashion) Classifier network for ten classes of human clothing images of the size 28x28 pixels.  
[Cifar 10](https://github.com/danielwilczak101/EasyNN/wiki/Cifar10) Classifier network for ten types of images varying from airplane, cat, dog, etc - 32x32 RGB images.

## To see more examples with many other datasets. Please visit our [wiki](https://github.com/danielwilczak101/EasyNN/wiki).
