## **EasyNN runs on Python 3.9.7 or greater.**
If you don't already have Python installed, you can install it here: https://www.python.org/downloads/

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
image = Preprocess("four.jpg").format(**format_options)

# Classify what the image is using the pretrained model.
print(model.classify(image))

# Show the image after it has been processed.
model.show(image)
```
### Output:
```bash
Downloading - four.jpg:
[################################] 1371/1371 - 00:00:00
4
```

### Image output:
<p align="center">
  <img width="400px" height="400px" src="https://danielwilczak101.github.io/EasyNN/images/example_four.png">
</p>

Since you now understand the basics, try one of our other trained models or create one with your data using our [untrained model section](https://github.com/danielwilczak101/EasyNN/wiki/Getting-Started-with-Untrained-Model).
