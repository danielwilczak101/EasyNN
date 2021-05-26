from examples.mnist import images, labels, show, model, preprocess_input

# Process my image so its size is  28x28 and in the correct greyscale values.
my_image = preprocess_input("examples/images/four.jpg")

# Model tells me what my image is.
model(my_image)

show(my_image)


# Model tells me what the image in thedataset is.
model(images[0])

show(images[0],labels[0])
