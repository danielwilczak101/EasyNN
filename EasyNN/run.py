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