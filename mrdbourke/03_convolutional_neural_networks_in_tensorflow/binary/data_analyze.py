import sys
sys.path.append('../../modules')
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

# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("data/pizza_steak/train/steak"))

print(f"num_steak_images_train: {num_steak_images_train}")

plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("data/pizza_steak/train/", "steak")
plt.subplot(1, 2, 2)
pizza_img = view_random_image("data/pizza_steak/train/", "pizza")

# Show original image and augmented image
plt.figure()
plt.subplot(1, 2, 1)
random_number = random.randint(0, 32) # we're making batches of size 32, so we'll get a random instance
plt.imshow(images[random_number])
plt.title(f"Original image")
plt.axis(False)
#plt.savefig('data/original_image.png', format='png')
plt.subplot(1, 2, 2)
#plt.figure()
plt.imshow(augmented_images[random_number])
plt.title(f"Augmented image")
plt.axis(False)
plt.savefig('data/augmented_image.png', format='png')
