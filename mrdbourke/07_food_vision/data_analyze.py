from common import *

print("")
print("# Check TensorFlow version (should be minimum 2.4.0+ but 2.13.0+ is better)")
print(f"TensorFlow version: {tf.__version__}")

print("")
print("# Add timestamp")
print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print("")
print("# Get all available datasets in TFDS")
datasets_list = tfds.list_builders()
print(datasets_list)
print(len(datasets_list))

print("")
print("# Set our target dataset and see if it exists")
target_dataset = "food101"
print(f"'{target_dataset}' in TensorFlow Datasets: {target_dataset in datasets_list}")

print("")
print("# Features of Food101 TFDS")
print(ds_info.features)

print("")
print("# Get class names")
class_names = ds_info.features["label"].names
print(class_names[:10])

print("")
print("# Take one sample off the training data")
train_one_sample = train_data.take(1) # samples are in format (image_tensor, label)
print("# What does one sample of our training data look like?")
print(train_one_sample)

print("")
print("# Output info about our training sample")
for image, label in train_one_sample:
  print(f"""
  Image shape: {image.shape}
  Image dtype: {image.dtype}
  Target class from Food101 (tensor form): {label}
  Class name (str form): {class_names[label.numpy()]}
        """)

print("")
print("# What does an image tensor from TFDS's Food101 look like?")
print(image)

print("")
print("# What are the min and max values?")
print(tf.reduce_min(image), tf.reduce_max(image))

print("")
print("# Plot an image tensor")
plt.imshow(image)
plt.title(class_names[label.numpy()]) # add title to image by indexing on class_names list
plt.axis(False)
plt.savefig("data/images/image_tensor.png")

print("")

