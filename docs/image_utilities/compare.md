Compare allows users to compare their own image to that of the dataset. For the example below we will be comparing our downloaded number 4 to other images in the dataset.

#### To use the compare utility:
```Python
from EasyNN.utilities import Preprocess, download
from EasyNN.utilities.image.compare import compare
from EasyNN.examples.mnist.number.data import dataset


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

# Compare our image to random dataset images.
compare(image,dataset=dataset)
```

#### Ouput:
```
Downloading - number_dataset.npz:
[################################] 11221/11221 - 00:00:00
Downloading - four.jpg:
[################################] 1371/1371 - 00:00:00
```

<p>
  <img width="450px" height="400px" src="https://danielwilczak101.github.io/EasyNN/images/compare_example.png">
</p>

