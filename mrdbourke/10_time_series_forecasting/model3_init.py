from common import *

HORIZON = 7
WINDOW_SIZE = 30

full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
print(len(full_windows), len(full_labels))

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
    windows=full_windows,
    labels=full_labels,
    test_split=0.2
)
print(len(train_windows), len(test_windows), len(train_labels), len(test_labels))

# Create model (same as model_1 except with different data input size)
model3 = Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(HORIZON)
], name="model3_dense")

model3.compile(
    loss="mae",
    optimizer=Adam()
)
                
