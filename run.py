from EasyNN.dataset.cifar.cifar10 import  dataset, model
import numpy as np

# Downloads dataset to computer
train_data,train_labels,test_data,test_labels = dataset

print(model(test_data[0]))

