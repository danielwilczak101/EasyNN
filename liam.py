from EasyNN.image.image import Image
import numpy as np
from EasyNN.image.mnist.show import show as mnist_show
from EasyNN.image.cifar.show import show as cifar_show

# RGB
image = Image("EasyNN/dataset/cifar/images/ship.png").format(resize=[32,32],flatten='F',rotate=1)

cifar_show(image)

# Grayscale
image = Image("EasyNN/dataset/mnist/number/images/three.jpg").format(grayscale=True,invert=True,process=True,contrast=30,resize=[28,28],rotate=3)

mnist_show(image)

# Create compare function.
# compare("three.jpg",train_data[0])