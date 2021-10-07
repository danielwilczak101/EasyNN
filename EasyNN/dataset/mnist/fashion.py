from EasyNN.dataset.mnist.show import *
from EasyNN.dataset.common import *
from EasyNN.model.network import *
from EasyNN.model.layer import *
from EasyNN.model.activation import *
from EasyNN.optimizer import *
from EasyNN.loss import *
from EasyNN.accuracy import *

model_filename = "fashion.model"
data_filename = "fashion.npz"
model_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/fashion_/fashion.model"
data_url = "https://github.com/danielwilczak101/EasyNN/raw/main/EasyNN/dataset/mnist/fashion_/fashion.npz"
labels = {
    0 :	'T-shirt/top',
    1 :	'Trouser',
    2 :	'Pullover',
    3 :	'Dress',
    4 :	'Coat',
    5 :	'Sandal',
    6 :	'Shirt',
    7 :	'Sneaker',
    8 :	'Bag',
    9 :	'Ankle boot',
}

model = Network()
# Add layers
model.add(Layer_Dense(784, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.show = show
model.labels = labels
# Finalize the model
model.finalize()


def __getattr__(name):
    if name == "model": 
        untrained_model = make_model(
            model,
            model_filename,
            data_filename,
            data_url,
            epochs=100
        )      
        return untrained_model

    if name == "trained_model":
        trained_model = make_trained_model(
        model,
        model_filename,
        model_url,
        data_filename,
        data_url,
    )
        return trained_model
        
    raise AttributeError(f"module {__name__} has no attribute {name}")
