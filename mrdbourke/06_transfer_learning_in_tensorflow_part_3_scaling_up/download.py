import subprocess

subprocess.run(['wget', 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py'])
subprocess.run(['wget', 'https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip'])
subprocess.run(['unzip', '101_food_classes_10_percent'])
subprocess.run(['mkdir', '-p', 'data'])
subprocess.run(['mv', '101_food_classes_10_percent', 'data'])
#unzip_data("101_food_classes_10_percent.zip")
subprocess.run(['rm', '101_food_classes_10_percent.zip'])
walk_through_dir("data/101_food_classes_10_percent")
