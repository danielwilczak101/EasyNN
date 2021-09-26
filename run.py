from EasyNN.dataset.mnist.fashion import trained_model, dataset, show

x1,y1,x2,y2 = dataset

user_image = x2[0]

show(user_image)

# Returns model
print(trained_model(user_image))