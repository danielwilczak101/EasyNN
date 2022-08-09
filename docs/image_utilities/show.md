Every trained model allows for a ``model.show()`` function that will output the data in a format that follows the dataset's form. A few examples can be seen below.

#### Dataset Example:
To learn more about the [MNIST handwritten dataset](https://github.com/danielwilczak101/EasyNN/wiki/MNIST-Numbers).
```Python
from EasyNN.examples.mnist.number.trained import model
from EasyNN.examples.mnist.number.data import dataset

images, labels = dataset

model.show(images[0])
```

#### Output:
<p>
  <img width="350px" height="300px" src="https://github.com/danielwilczak101/EasyNN/blob/media/images/five_example_show.png">
</p>

#### Downloaded Example:
This functionality can also be paired up with downloaded images that are properly sized.

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

#### Output:
<p>
  <img width="350px" height="300px" src="/images/dress_example.png">
</p>
