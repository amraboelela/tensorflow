import sys
sys.path.append('../modules')
from helper_functions import *

# Start by using the 20k dataset
data_dir = "data/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

filenames = [data_dir + filename for filename in os.listdir(data_dir)]

train_lines = get_lines(data_dir+"train.txt")

print("# Get data from file and preprocess it")
#%%time
train_samples = preprocess_text_with_line_numbers(data_dir + "train.txt")
val_samples = preprocess_text_with_line_numbers(data_dir + "dev.txt") # dev is another name for validation set
test_samples = preprocess_text_with_line_numbers(data_dir + "test.txt")
     
