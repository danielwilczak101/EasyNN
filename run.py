from EasyNN.dataset.mnist.number import trained_model, dataset, show

# Downloads dataset to computer
train_data,train_labels,test_data,test_labels = dataset

# Grab a training data image
image = test_data[0]

# Uses the EasyNN train model on an example test image.
print(trained_model(image))

# Show the image
show(image, "image")