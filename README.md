![](https://raw.githubusercontent.com/danielwilczak101/EasyNN/media/images/readme_logo.png)

# EasyNN - Neural Networks made Easy
EasyNN is a python package designed to provide an easy-to-use Neural Network. The package is designed to work right out of the box, while also allowing the user to customize features as they see fit. 

## Check out our [wiki](https://github.com/danielwilczak101/EasyNN/wiki) for more information.

### Current working code:
```Python
from EasyNN.dataset.mnist import model, dataset, show

trainx, trainy, xtest, ytrain = dataset

show(xtest[1])

model.predictions(xtest[1])
model.predict(xtest[1])
```


### Goal code:
```Python
from EasyNN.mnist import model, dataset, compare, preprocess 

compare(user_image,dataset)
model.train(dataset)

# If you need to preprocess
user_image = preprocess(user_image)

model(user_image)
```

### Ouput:
```

# Box with 3 images:
  First: User Image
  Second: Dataset Image
  Third: Pre-processed image
  
 # Model training loading bar
 
 # Model prediction of what the image is.
 
```

### Image Output:
```
```

### Future goals for non known datasets
```Python
from EasyNN.model import model

xtrain = "My images"
ytrain = "My labels"

model.dataset = xtrain, ytrain

model(xtrain[0])
```
