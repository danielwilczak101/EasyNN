<p align="center">
  <img src="/images/cifar10_example.png">
</p>

## CIFAR-10:
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

## Model Output Dictionary:
Used to translate the neural networks probability out to a string for human consumption.
```Python
model.labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}
```

### Model:
```Python
from EasyNN.examples.cifar10.trained import model

# Classify the an image in the dataset
print(model.classify(image))
```

### Dataset Example:
```Python
from EasyNN.examples.cifar10.trained import model
from EasyNN.examples.cifar10.data import dataset

images, labels = dataset

# Classify what the second image is in the dataset.
print(model.classify(images[1]))
```

### Dataset example output:
```
Downloading - cifar10_parameters.npz:
[################################] 25541/25541 - 00:00:05
Downloading - cifar10_structure.pkl:
[################################] 29834/29834 - 00:00:05
Downloading - cifar10_dataset.npz:
[################################] 165539/165539 - 00:01:04
ship
```


### Full example:
More info can be found about [converting images](https://github.com/danielwilczak101/EasyNN/wiki/Image-Utility) in the utilities section.
```Python
from EasyNN.examples.cifar10.trained import model
from EasyNN.utilities import Preprocess, download

# Download an example image.
download("ship.png","https://bit.ly/31jyerQ")

format_options = dict(
    resize=(32, 32),
    rotate=1,
    flatten='F'
)

# Converting your image into the correct format for the mnist number dataset.
image = Preprocess("ship.png").format(**format_options)

# Show the image after it has been processed.
model.show(image)

# Classify what the image is using the pretrained model.
print(model.classify(image))
```
### Output:
```bash
Downloading - cifar10_parameters.npz:
[################################] 25541/25541 - 00:00:05
Downloading - cifar10_structure.pkl:
[################################] 26754/26754 - 00:00:05
Downloading - cifar10_dataset.npz:
[################################] 165539/165539 - 00:00:51
ship
```
### Image output:
Shows what the image from the internet looks like and what it becomes after being resized and processed.
<p>
  <img src="images/image_compare.png">
</p>






