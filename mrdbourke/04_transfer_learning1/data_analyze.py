from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

# Walk through 10 percent data directory and list number of files
for dirpath, dirnames, filenames in os.walk("data/10_food_classes_10_percent"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
