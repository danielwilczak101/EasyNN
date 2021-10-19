from EasyNN.model import Network, Normalize, ReLU, LogSoftMax
from EasyNN.examples.mnist.number import labels, show
import numpy as np
import pickle

# Create the mnist model.
model = Network(
    Normalize(1e-3),
    256, ReLU,
    128, ReLU,
    10, LogSoftMax
)

# Establish the models labels
model.labels = labels

# Use the pre-created show function to 
model.show = show

# Save the model as a pick file to be used later.
# Its important to use binary mode
dbfile = open('number_model.pkl', 'wb')
    
# source, destination
pickle.dump(model, dbfile)                     
dbfile.close()