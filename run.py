from EasyNN.dataset.mnist.number import  dataset, preprocess, show
import numpy as np
from PIL import Image, ImageOps

# Downloads dataset to computer
train_data,train_labels,test_data,test_labels = dataset

user_img = np.array(Image.open("three.jpg").convert('L'))

show(user_img)

