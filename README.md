![](https://raw.githubusercontent.com/danielwilczak101/EasyNN/media/images/readme_logo.png)

# EasyNN - Neural Networks made Easy

This project is still in work and does not have a production version yet.

EasyNN is a python package designed to provide an easy-to-use Neural Network. The package is designed to work right out of the box, while also allowing the user to customize features as they see fit. 

## Check out our [wiki](https://github.com/danielwilczak101/EasyNN/wiki) for more information.

Current example of working code
```Python
from EasyNN import mnist

#model = NeuralNetwork()

#model.train()

# This will take a bit if its your first time running since it has to
# download 50MB sqlite3 file.
mnist = mnist()

train = mnist.training
test  = mnist.testing

mnist.show_image(train[0])

# model(image)
# tell me what the probabilities are
```

## Ouput:
```
Downloading MNIST dataset:
[################################] 56148/56149 - 00:02:19
Getting 50,000 training examples. This may take a minute:
100%|███████████████████████████████████| 60000/60000 [00:14<00:00, 4250.58it/s]
Getting 10,000 testing examples. This may take a minute:
100%|███████████████████████████████████| 10000/10000 [00:02<00:00, 4350.04it/s]
```

