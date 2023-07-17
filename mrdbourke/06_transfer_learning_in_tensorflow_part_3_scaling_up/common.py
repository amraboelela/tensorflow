import sys
sys.path.append('../modules')
from helper_functions import *

train_dir = "data/101_food_classes_10_percent/train/"
test_dir = "data/101_food_classes_10_percent/test/"

# Setup data inputs
IMG_SIZE = (224, 224)
train_data_all_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                label_mode="categorical",
                                                                                image_size=IMG_SIZE)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                shuffle=False) # don't shuffle test data for prediction analysis
class_names = test_data.class_names

imagePath = "/home/amr/webserver/HaneinWebserver/Public/images"
print("")
print("")

