from common import *

print("train_df.head()")
print(str(train_df.head()))
print("")

# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility
print("train_df_shuffled.head()")
print(str(train_df_shuffled.head()))
print("")

# The test data doesn't have a target (that's what we'd try to predict)
print("test_df.head()")
print(str(test_df.head()))
print("")

# How many examples of each class?
print("train_df.target.value_counts()")
print(str(train_df.target.value_counts()))
print("")

