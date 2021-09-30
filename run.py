from EasyNN.dataset.mnist.number import trained_model, dataset, trained_model
import numpy as np

# Downloads dataset to computer
train_data,train_labels,test_data,test_labels = dataset

trained_model(train_data[0])