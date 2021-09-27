from EasyNN.dataset.mnist.number import trained_model, dataset, show

# Downloads dataset to computer
train_data,train_labels,test_data,test_labels = dataset

user_image = test_data[0]

# Uses the EasyNN train model on an example test image.
print(trained_model(user_image))

show(user_image, "image")  # Shows image as matplotlib graph
show(user_image, "array")  # Shows image as numpy array print out to proper width.

