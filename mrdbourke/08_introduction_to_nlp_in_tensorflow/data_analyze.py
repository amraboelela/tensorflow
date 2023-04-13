from common import *

print(str(train_df.head()))

# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility
print(str(train_df_shuffled.head()))

# The test data doesn't have a target (that's what we'd try to predict)
print(str(test_df.head()))

