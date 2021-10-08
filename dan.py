from EasyNN.image.image import Image
import numpy as np

# RGB
image = Image("EasyNN/dataset/cifar/images/ship.png").format(resize=[32,32],flatten='F',rotate=1)

# Grayscale
image = Image("EasyNN/dataset/mnist/number/images/three.jpg").format(grayscale=True,invert=True,process=True,contrast=30,resize=[28,28],rotate=3)

#TODO

# Fix so the model gets the show function 
#trained_model.show(image)

# Create compare function.
# compare("three.jpg",train_data[0])
