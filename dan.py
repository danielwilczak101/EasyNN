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
create_image_dataset(name="Animal", path="./images/",
                     size=(28, 28), greyscale=True)

# Load the data
dataset = load("Animal_dataset.npz")

# Setup variables.
images, labels = dataset

# Tell us the size so we know everything is there.
print(images.shape)
print(labels.shape)

# Print the labels using there names and not the numbers they were entered.
names = {
    0: "dog",
    1: "cat"
}

for index, label in enumerate(labels):
    print(f"Image #{index} is a {names[label]}")
