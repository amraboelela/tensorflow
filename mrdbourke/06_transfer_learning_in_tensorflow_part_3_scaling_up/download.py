import subprocess
from helper_functions import *

subprocess.run(['wget', 'https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip'])
#subprocess.run(['unzip', '101_food_classes_10_percent'])
unzip_data("101_food_classes_10_percent.zip")
subprocess.run(['rm', '101_food_classes_10_percent.zip'])
walk_through_dir("101_food_classes_10_percent")
