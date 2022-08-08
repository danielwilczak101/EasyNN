Create a dataset from images in a folders with folder names as labels.

### Required folder structure:
The files need to be structured in this way.
```
Images:
    ├── cat <- Folder and label to the images.
    │   ├── 0.jpg <- Image of cat
    │   ├── 1.jpg
    └── dog <- Folder and label to the images.
        ├── 0.jpg <- Image of dog
        ├── 1.jpg <- Image of dog
```

### Example
This example assumes you have the file structured as described above.
```Python
from EasyNN.utilities.data.load import load
from EasyNN.utilities.create import create_image_dataset

# Take the images from the folders and create the dataset 
# file named Animal with size image sizes of (28,28 pixels) 
# and make it greyscale to save some space.
create_image_dataset(name="Animal", path="./images/", size=(28,28), greyscale=True)

# Load the data
dataset = load("Animal_dataset.npz")

# Setup variables.
images, labels = dataset

# Tell us the size so we know everything is there.
print(images.shape)
print(labels.shape)
```

### Full Usable Example:
Using the [EasyNN google scrape feature](https://github.com/danielwilczak101/EasyNN/wiki/Scraping-Google-Images) we'll download 10 images of dogs and 10 cat images and transform them into a dataset.
```Python
from EasyNN.utilities.data.load import load
from EasyNN.utilities.create import create_image_dataset
from EasyNN.utilities.scrape import scrape_google

# Get some images from google so we can make a dataset.
    # Make sure to check out the documentation on scraping google to get this work ing
    # - https://github.com/danielwilczak101/EasyNN/wiki/Scraping-Google-Images

scrape_google("dog", count=10)
scrape_google("cat", count=10)

# Take the images from the folders and create the dataset 
# file named Animal with size image sizes of (28,28 pixels) 
# and make it greyscale to save some space.
create_image_dataset(name="Animal", path="./images/", size=(28,28), greyscale=True)

# Load the data
dataset = load("Animal_dataset.npz")

# Setup variables.
images, labels = dataset

# Tell us the size so we know everything is there.
print(images.shape)
print(labels.shape)
```