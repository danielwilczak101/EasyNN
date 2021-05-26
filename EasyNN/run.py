from examples.mnist import images, labels, show, model, preprocess_input

#print(images[0])
#print(labels[0])


my_image = preprocess_input("four.jpg")

# Model tells me what the image is.
model(images[2])

show(my_image)
