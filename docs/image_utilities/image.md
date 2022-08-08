Images are a huge part of neural networks. Classification of images is a huge resource for many projects requiring some form of computer vision. EasyNN has created a image class to make converting your image into the proper form to match the dataset your using to classify. Let's get to some code.

Import to get all the functionality.
```Python
from EasyNN.utilities.image.preprocess import image
```

## Simple full example
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

# Show function is tied to a model.
model.show(image)
```

## Image Parameters
These are the current attributes that the image class allows you to modify.  

#### grayscale=True / False  
The grayscale parameter will change the image into grayscale pixels that have a rand from 0-255 of intensity.  
  
#### invert=True / False  
Inverted the colors of the image. If the pixel is white its now black for grayscale values.  

#### process=True / False   
Preprocess allows for noise to be removed from your image by taking the average of all the pixels and remove up to that threshold. 

Example of the pixels:  
```
Average is: 207/9 = 23

Before          After:
[30,10,30],     [7,0,7],
[25,25,25],     [3,3,2],
[25,10,30]      [2,0,7]
```

Image example:
```
SHOW COMPARE IMAGES
```

#### contrast= 0-255  
Contrast will remove noise but with an amount. The higher the number the more the image will remove pixel values.   
  
#### resize=(28, 28) - (height,width) of pixels.    
Change the image to a desired height and width.  

#### rotate=1-4  
Rotate the image. Number represents rotation by 90 degrees clockwise.  




