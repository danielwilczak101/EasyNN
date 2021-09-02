![](https://raw.githubusercontent.com/danielwilczak101/EasyNN/media/images/readme_logo.png)

# EasyNN - Neural Networks made Easy
EasyNN is a python package designed to provide an easy-to-use Neural Network. The package is designed to work right out of the box, while also allowing the user to customize features as they see fit. 

## Check out our [wiki](https://github.com/danielwilczak101/EasyNN/wiki) for more information.

Current example of working code
```Python
from EasyNN import mnist

# Dowload the MNIST dataset.
mnist = mnist()

model = NeuralNetwork()
model.train()

mnist.show_image(train[0])

# If you want to show the images data
print(train[0])

# Tell me what the model thinks it is.
model(image)

```

## Ouput:
```
Downloading MNIST dataset:
[################################] 56148/56149 - 00:02:19
Getting 50,000 training examples. This may take a minute:
100%|███████████████████████████████████| 60000/60000 [00:14<00:00, 4250.58it/s]
Getting 10,000 testing examples. This may take a minute:
100%|███████████████████████████████████| 10000/10000 [00:02<00:00, 4350.04it/s]

[5, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... ]]

[5]
```

## Image Output:
![](https://github.com/danielwilczak101/EasyNN/blob/media/images/image_preview.png)
