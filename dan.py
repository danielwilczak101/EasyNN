from EasyNN.dataset.mnist.number import trained_model as model

image = model.testing.data[0]

model.show(image)

print(model.classify(image))
print(model.accuracy())


