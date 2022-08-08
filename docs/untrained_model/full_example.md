This full example creates the an [XOR model](https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7), trains the model and saves it.
```Python
from EasyNN.model import Network, ReLU, LogSoftMax

import EasyNN.callbacks as cb
import numpy as np

# Setup the XOR dataset.
data = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
] * 100, dtype=int)

labels = np.array([0,1,1,0] * 100, dtype=int)

# Create the Neural Network Model.
model = Network(
    16, ReLU,
    2, LogSoftMax,
)

# Set your models data
model.training.data = (data, labels)

# Set the models labels
model.labels = {
    0: 0,
    1: 1
}

# Change to a small learning rate.
model.optimizer.lr = 0.01

 
# Set when to terminate point. Training will end once your
# validation accuracy hits above 90% two times.
model.callback(
    cb.ReachValidationAccuracy(limit=0.90, patience=2),
)

# Print the accuracy and iteration count every 10 iterations.
model.print.on_validation_start(iteration=True,accuracy=True)
model.print.on_training_start(iteration=True, frequency=10)

# Always at the end of your setup
model.train()

# Save your model so that you can use it later.
model.save("xor")
```

#### Output:
```
Iteration: 0
Iteration: 10
Iteration: 20
Iteration: 30
Iteration: 40
Iteration: 44, Validation Accuracy: 0.76171875
Iteration: 50
Iteration: 60
Iteration: 70
Iteration: 80
Iteration: 89, Validation Accuracy: 1.0
Iteration: 90
Iteration: 100
Iteration: 110
Iteration: 120
Iteration: 130
Iteration: 135, Validation Accuracy: 1.0
Parameters saved.
```


### Loading Trained/Saved model example:
This example is using the [XOR model](https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7) example above.
```Python
from EasyNN.model import Model
import numpy as np

# Load the trained model we saved.
model = Model.load("xor")

# Classify an example.
print(model.classify(np.array([0,1])))
```

#### Output:
```
1
```
