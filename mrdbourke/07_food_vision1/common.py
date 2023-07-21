import sys
sys.path.append('../modules')
from helper_functions import *

# List available datasets
datasets_list = tfds.list_builders() # get all available datasets in TFDS
print("food101" in datasets_list) # is the dataset we're after available?
