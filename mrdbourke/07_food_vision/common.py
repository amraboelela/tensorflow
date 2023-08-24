import sys
sys.path.append('../modules')
from helper_functions import *

print("")
print("# Load in the data (takes about 5-6 minutes in Google Colab)")
raw_train_data, ds_info = read_dataset("raw_train_data")
raw_test_data, ds_info = read_dataset("raw_test_data")
if raw_train_data is None or raw_test_data is None:
  (raw_train_data, raw_test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                             split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                             shuffle_files=True, # shuffle files on download?
                                             as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                             with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)
  save_dataset(raw_train_data, ds_info, "raw_train_data")
  save_dataset(raw_test_data, ds_info, "raw_test_data")

class_names = ds_info.features["label"].names

# Map preprocessing function to training data (and paralellize)
train_data = raw_train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map prepreprocessing function to test data
test_data = raw_test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

download("https://storage.googleapis.com/ztm_tf_course/food_vision/07_efficientnetb0_feature_extract_model_mixed_precision.zip")

