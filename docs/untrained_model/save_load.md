Saving and loading your trained model is critical. No one wants to spend hours training to just use there model once. EasyNN handles saving and loading in a few lines.

## Save:
Once your model has been trained. This is all you will need to save. 
```Python
model.save("name")
```

Saving the model will output two files. The **parameters file** are all the models optimized numbers. The **structure** is a pickle file that stores most of the code necessary to run a model.
```Bash
name_parameters.npz
name_structure.pkl
```

```Python
from EasyNN.examples.cifar10.trained import model
from EasyNN.examples.cifar10.data import dataset

images, labels = dataset

# Show the image
model.show(images[2])
```



Click here to see a [Save full example](https://github.com/danielwilczak101/EasyNN/wiki/Untrained-Full-Example).

## Loading Model:
To load a model its only two lines. Importing the base model class and loading your saved model by using its name.
```Python
from EasyNN.model import Model

model = Model.load("name")
```

Click here to see a [Load full example](https://github.com/danielwilczak101/EasyNN/wiki/Untrained-Full-Example#loading-trainedsaved-model-example).

