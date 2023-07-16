import sys
sys.path.append('../../modules')
from helper_functions import *
from common import *

# Walk through 10_food_classes directory and list number of files
for dirpath, dirnames, filenames in os.walk("data/10_food_classes_all_data"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
