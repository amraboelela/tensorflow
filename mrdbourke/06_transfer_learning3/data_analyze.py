from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print("")
print("# How many images/classes are there?")
walk_through_dir("data/101_food_classes_10_percent")
