from EasyNN.examples.mnist.fashion.untrained import model
from EasyNN.utilities import Preprocess, download

try:
    model.train()
except KeyboardInterrupt:
    model.save("fashion")
