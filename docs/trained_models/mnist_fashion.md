<p align="center">
  <img src="images/fashion_mnist_example.jpg">
</p>

## MNIST Fashion:
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. 

## Dataset Labels:
Used to translate the neural networks probability out to a string for human consumption.
```Python
model.labels = {
    0 :	'T-shirt/top',
    1 :	'Trouser',
    2 :	'Pullover',
    3 :	'Dress',
    4 :	'Coat',
    5 :	'Sandal',
    6 :	'Shirt',
    7 :	'Sneaker',
    8 :	'Bag',
    9 :	'Ankle boot',
}
```

### Model:
```Python
from EasyNN.examples.mnist.fashion.trained import model

# Classify the an image in the dataset
print(model.classify(image))
```

### Dataset Example:
```Python
from EasyNN.examples.mnist.fashion.trained import model
from EasyNN.examples.mnist.fashion.data import dataset

images, labels = dataset

# Classify what the second image is in the dataset.
print(model.classify(images[0]))

# Show the image.
model.show(images[0])
```

### Dataset example output:
```
Downloading - fashion_parameters.npz:
[################################] 4765/4765 - 00:00:00
Downloading - fashion_structure.pkl:
[################################] 13831/13831 - 00:00:00
Downloading - fashion_dataset.npz:
[################################] 30147/30147 - 00:00:00
Ankle boot
```


### Full example:
More info can be found about [converting images](https://github.com/danielwilczak101/EasyNN/wiki/Image-Utility) in the utilities section.
```Python
from EasyNN.examples.mnist.fashion.trained import model
from EasyNN.utilities import Preprocess, download

download("dress.jpg","https://bit.ly/3b7rsXF")

format_options = dict(
    grayscale=True,
    invert=True,
    process=True,
    contrast=5,
    resize=(28, 28),
    rotate=0,
)

# Converting your image into the correct format for the mnist fashion dataset.
image = Preprocess("dress.jpg").format(**format_options)

# Show the image after it has been processed.
model.show(image)

# Classify what the image is using the pretrained model.
print(model.classify(image))
```
### Output:
```bash
Downloading - fashion_parameters.npz:
[################################] 4765/4765 - 00:00:00
Downloading - fashion_structure.pkl:
[################################] 13831/13831 - 00:00:00
Downloading - fashion_dataset.npz:
[################################] 30147/30147 - 00:00:00
Downloading - dress.jpg:
[################################] 25/25 - 00:00:00
Dress
```
### Image output:
<p>
  <img width="350px" height="300px" src="images/dress_example.png">
</p>