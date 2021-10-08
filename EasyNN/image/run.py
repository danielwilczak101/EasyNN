from EasyNN.dataset.cifar.cifar10 import trained_model
from EasyNN.preprocess.image import Image
import numpy as np
# Downloads dataset to computer
train_labels,train_data,test_data,test_labels = trained_model.dataset



# RGB
#image = Image("ship.png").format(resize=[32,32],flatten='F',rotate=1)

# Grayscale
#image = Image("three.jpg").format(grayscale=True,invert=True,process=True,contrast=30,resize=[28,28],rotate=3)

# Show the image
#trained_model.show(image)

#print(trained_model.classify(image))

# compare("three.jpg",train_data[0])

x = np.concatenate((train_data , test_data),axis=0)
y = np.concatenate((train_labels , test_labels), axis=0)

np.savez_compressed('combined_cifar10.npz', x=x, y=y)