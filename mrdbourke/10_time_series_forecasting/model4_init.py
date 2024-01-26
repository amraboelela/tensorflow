from common import *

# Create windowed dataset
full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
print(len(full_windows), len(full_labels))

# Create train/test splits
train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
    full_windows,
    full_labels
)
print(len(train_windows), len(test_windows), len(train_labels), len(test_labels))

print()
print("# Check data sample shapes")
print(train_windows[0].shape) # returns (WINDOW_SIZE, )

print()
print("# Before we pass our data to the Conv1D layer, we have to reshape it in order to make sure it works")
x = tf.constant(train_windows[0])
expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1)) # add an extra dimension for timesteps
print(f"Original shape: {x.shape}") # (WINDOW_SIZE)
print(f"Expanded shape: {expand_dims_layer(x).shape}") # (WINDOW_SIZE, input_dim)
print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")

# Create model
model4 = Sequential([
        # Create Lambda layer to reshape inputs, without this layer, the model will error
        layers.Lambda(lambda x: tf.expand_dims(x, axis=1)), # resize the inputs to adjust for window size / Conv1D 3D input requirements
        layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
        layers.Dense(HORIZON)
    ],
    name="model4_conv1D"
)

# Compile model
model4.compile(
    loss="mae",
    optimizer=Adam()
)
                
