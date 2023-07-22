import sys
sys.path.append('../../../modules')
from helper_functions import *

# The data has already been sorted into training and test sets for us
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
