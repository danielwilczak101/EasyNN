<p align="center">
  <img src="images/MnistExamples.png">
</p>


## MNIST Number:
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

### Model:
```Python
from EasyNN.examples.mnist.number.trained import model

# Classify the an image in the dataset
print(model.classify(image))
```

### Dataset Example:
```Python
from EasyNN.examples.mnist.number.trained import model
from EasyNN.examples.mnist.number.data import dataset

images, labels = dataset

# Classify what the second image is in the dataset.
print(model.classify(images[0]))

# Show the image.
model.show(images[0])
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
  <img width="400px" height="400px" src="images/example_three.png">
</p>