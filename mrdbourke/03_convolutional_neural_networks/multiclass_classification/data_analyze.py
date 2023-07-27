from common import *

# Walk through 10_food_classes directory and list number of files
for dirpath, dirnames, filenames in os.walk("data/10_food_classes_all_data"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

img = view_random_image(target_dir=train_dir,
                        target_class=random.choice(class_names)) # get a random class name
