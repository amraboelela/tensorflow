import sys
sys.path.append('../modules')
from helper_functions import *

#img = view_random_image(target_dir="data/pizza_steak/train/",
#                        target_class="steak")
#print(img)
#print(img.shape)
#print(img/255.)

plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("data/pizza_steak/train/", "steak")
plt.subplot(1, 2, 2)
pizza_img = view_random_image("data/pizza_steak/train/", "pizza")
