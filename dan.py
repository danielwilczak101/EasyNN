from EasyNN.dataset.mnist.number import trained_model
from EasyNN.utilities.image.image import Image

# Example code to run with trained model and images.
image = Image("EasyNN/dataset/mnist/number/images/four.jpg").format(grayscale=True,invert=True,process=True,contrast=30,resize=[28,28],rotate=3)

print(trained_model.classify(image))

trained_model.show(image)

