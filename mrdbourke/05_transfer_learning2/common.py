import sys
sys.path.append('../modules')
from helper_functions import *

#remove_subdirectories_with_leaf("data/transfer_learning", "train") # Keep only validate directories

download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip")

# Create training and test directories
train_dir_10_percent = "data/10_food_classes_10_percent/train/"
test_dir = "data/10_food_classes_10_percent/test/"

# Create data inputs
IMG_SIZE = (224, 224) # define image size

train_data_10_percent = image_dataset_from_directory(
    directory=train_dir_10_percent,
    image_size=IMG_SIZE,
    label_mode="categorical", # what type are the labels?
    batch_size=32
) # batch_size is 32 by default, this is generally a good number

test_data = image_dataset_from_directory(
    directory=test_dir,
    image_size=IMG_SIZE,
    label_mode="categorical"
)

download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip")

# Create training and test dirs
train_dir_1_percent = "data/10_food_classes_1_percent/train/"
test_dir = "data/10_food_classes_1_percent/test/"

train_data_1_percent = image_dataset_from_directory(
    train_dir_1_percent,
    label_mode="categorical",
    batch_size=32, # default
    image_size=IMG_SIZE
)
    
test_data = image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    image_size=IMG_SIZE
)

download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip")

# Setup data directories
train_dir_all_data = "data/10_food_classes_all_data/train/"
test_dir = "data/10_food_classes_all_data/test/"

train_data_all = image_dataset_from_directory(
    train_dir_all_data,
    label_mode="categorical",
    image_size=IMG_SIZE
)

# Note: this is the same test dataset we've been using for the previous modelling experiments
test_data = image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    image_size=IMG_SIZE
)
                                                            
