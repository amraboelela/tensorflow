from helper_functions import *
    
subprocess.run(['mkdir', '-p', 'data'])
download_resource('food_vision/101_food_classes_10_percent')
walk_through_dir("data/101_food_classes_10_percent")
print("")

