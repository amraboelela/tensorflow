from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")
print(f"TensorFlow version: {tf.__version__}")

print("")
print("# Walk through 10 percent data directory and list number of files")
walk_through_dir("data/10_food_classes_10_percent")

print("")
print("# Check the training data datatype")
print(train_data_10_percent)

print("")
print("# Check out the class names of our dataset")
print(train_data_10_percent.class_names)

print("")
print("# See an example batch of data")
for images, labels in train_data_10_percent.take(1):
  print(images, labels)

# Define input tensor shape (same number of dimensions as the output of efficientnetb0)
input_shape = (1, 4, 4, 3)

# Create a random tensor
tf.random.set_seed(42)
input_tensor = tf.random.normal(input_shape)
print(f"Random input tensor:\n {input_tensor}\n")

# Pass the random tensor through a global average pooling 2D layer
global_average_pooled_tensor = GlobalAveragePooling2D()(input_tensor)
print(f"2D global average pooled random tensor:\n {global_average_pooled_tensor}\n")

# Check the shapes of the different tensors
print(f"Shape of input tensor: {input_tensor.shape}")
print(f"Shape of 2D global averaged pooled input tensor: {global_average_pooled_tensor.shape}")

print("")
print("# This is the same as GlobalAveragePooling2D()")
print(tf.reduce_mean(input_tensor, axis=[1, 2]), "# average across the middle axes")

print("")
print("# Doing the same but for GlobalMaxPool2D()")
global_max_pooled_tensor = GlobalMaxPool2D()(input_tensor)
print(global_max_pooled_tensor)

print("")
print("# Walk through 1 percent data directory and list number of files")
walk_through_dir("data/10_food_classes_1_percent")


target_class = random.choice(train_data_1_percent.class_names) # choose a random class
target_dir = "data/10_food_classes_1_percent/train/" + target_class # create the target directory
random_image = random.choice(os.listdir(target_dir)) # choose a random image from target directory
random_image_path = target_dir + "/" + random_image # create the choosen random image path
img = mpimg.imread(random_image_path) # read in the chosen target image
plt.imshow(img) # plot the target image
plt.title(f"Original random image from class: {target_class}")
plt.axis(False) # turn off the axes
plt.savefig('data/images/random_image.png', format='png')
    
# Augment the image
augmented_img = data_augmentation(tf.expand_dims(img, axis=0)) # data augmentation model requires shape (None, height, width, 3)
plt.figure()
augmented_img = tf.cast(augmented_img, dtype=tf.int32)
plt.imshow(tf.squeeze(augmented_img)) #/255.) # requires normalization after augmentation, or converting to int32
plt.title(f"Augmented random image from class: {target_class}")
plt.axis(False)
plt.savefig('data/images/augmented_random_image.png', format='png')

print("")
print("# How many images are we working with now?")
walk_through_dir("data/10_food_classes_all_data")
