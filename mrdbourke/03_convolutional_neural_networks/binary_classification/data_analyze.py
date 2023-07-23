from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print("")
print("# Walk through pizza_steak directory and list number of files")
for dirpath, dirnames, filenames in os.walk("data/pizza_steak"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
  
print("")
print("# Another way to find out how many images are in a file")
num_steak_images_train = len(os.listdir("data/pizza_steak/train/steak"))

print(num_steak_images_train)

print("")
print("# Get the class names (programmatically, this is much more helpful with a longer list of classes)")
data_dir = pathlib.Path("data/pizza_steak/train/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)

print("")
print("# View a random image from the training dataset")
img = view_random_image(target_dir="data/pizza_steak/train/",
                        target_class="steak")
                 
print("")
print("# View the img (actually just a big array/tensor)")
print(img)

print("")
print("# View the image shape")
print(img.shape) # returns (width, height, colour channels)

print("")
print("# Get all the pixel values between 0 & 1")
print(img/255.)

# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("data/pizza_steak/train/steak"))

print(f"num_steak_images_train: {num_steak_images_train}")

plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("data/pizza_steak/train/", "steak")
plt.subplot(1, 2, 2)
pizza_img = view_random_image("data/pizza_steak/train/", "pizza")

print("")
print("# Get a sample of the training data batch")
images, labels = train_data.next() # get the 'next' batch of images/labels
print(len(images), len(labels))
print("")
print("# View the first batch of labels")
print(labels)
print("")
print("# Get the first two images")
print(images[:2], images[0].shape)

augmented_images, augmented_labels = train_data_augmented.next() # Note: labels aren't augmented, they stay the same
print(f"len(images): {len(images)} len(labels): {len(labels)}")

# Get the first two images
print(f"{images[:2]}, {images[0].shape}")

# Check lengths of training and test data generators
print(f"{len(train_data)}, {len(test_data)}")

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
plt.savefig('data/images/augmented_image.png', format='png')
