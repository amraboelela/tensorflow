import sys
sys.path.append('../modules')
from helper_functions import *

img = view_random_image(target_dir="data/pizza_steak/train/",
                        target_class="steak")
print(img)
print(img.shape)
print(img/255.)
