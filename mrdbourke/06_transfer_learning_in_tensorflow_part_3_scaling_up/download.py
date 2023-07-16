from helper_functions import *
    
subprocess.run(['mkdir', '-p', 'data'])
download("https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip")
walk_through_dir("data/101_food_classes_10_percent")
print("")

