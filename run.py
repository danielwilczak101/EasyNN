from EasyNN.dataset.mnist.fashion import trained_model
from EasyNN.preprocess.image import Image

# Downloads dataset to computer
train_data,train_labels,test_data,test_labels = trained_model.dataset

# RGB
#image = Image("ship.png").format(resize=[32,32],flatten='F',rotate=1)

# Grayscale
image = Image("three.jpg").format(grayscale=True,invert=True,process=True,contrast=30,resize=[28,28],rotate=3)

# Show the image
trained_model.show(image)

print(trained_model.classify(image))

# compare("three.jpg",train_data[0])
