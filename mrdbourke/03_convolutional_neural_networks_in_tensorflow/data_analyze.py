import sys
sys.path.append('../modules')
from helper_functions import *
from common import *

# Get data batch samples
images, labels = train_data.next()
augmented_images, augmented_labels = train_data_augmented.next() # Note: labels aren't augmented, they stay the same
print(f"len(images): {len(images)} len(labels): {len(labels)}")

# Get the first two images
print(f"{images[:2]}, {images[0].shape}")

# Check lengths of training and test data generators
print(f"{len(train_data)}, {len(test_data)}")

plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("data/pizza_steak/train/", "steak")
plt.subplot(1, 2, 2)
pizza_img = view_random_image("data/pizza_steak/train/", "pizza")
