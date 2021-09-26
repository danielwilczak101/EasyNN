from EasyNN.dataset.mnist.fashion import trained_model, show, dataset

x1,y1,x2,y2 = dataset

user_image = x2[1]

show(user_image)

# Returns model
print(trained_model(user_image))