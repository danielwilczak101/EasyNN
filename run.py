from EasyNN.dataset.mnist.number import model, show, dataset

x1,y1,x2,y2 = dataset

user_image = x2[4]

show(user_image)

# Returns model
print(model(user_image))


