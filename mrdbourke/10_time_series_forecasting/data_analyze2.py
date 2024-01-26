from common2 import *

print()
print("# Get date indexes for when to add in different block dates")
print(f"bitcoin_prices.index[0]: {bitcoin_prices.index[0]}")
print(block_reward_2_days, block_reward_3_days)

print()
print("# Set values of block_reward column (it's the last column hence -1 indexing on iloc)")
print(bitcoin_prices_block.head())

# Plot the block reward/price over time
# Note: Because of the different scales of our values we'll scale them to be between 0 and 1.
scaled_price_block_df = pd.DataFrame(
    minmax_scale(bitcoin_prices_block[["Price", "block_reward"]]), # we need to scale the data first
    columns=bitcoin_prices_block.columns,
    index=bitcoin_prices_block.index
)
plt.figure(figsize=(10, 7))
scaled_price_block_df.plot(figsize=(10, 7))
plt.savefig('data/images/scaled_price_block_df.png', format='png')

print()
print("# Add windowed columns")
print(bitcoin_prices_windowed.head(10))

print()
print("# Let's create X & y, remove the NaN's and convert to float32 to prevent TensorFlow errors")
print(X.head())

print()
print("# View labels")
print(y.head())

print()
print("# Make train and test sets")
print(len(X_train), len(y_train), len(X_test), len(y_test))


# Set up dummy NBeatsBlock layer to represent inputs and outputs
dummy_nbeats_block_layer = NBeatsBlock(
    input_size=WINDOW_SIZE,
    theta_size=WINDOW_SIZE+HORIZON, # backcast + forecast
    horizon=HORIZON,
    n_neurons=128,
    n_layers=4
)

# Create dummy inputs (have to be same size as input_size)
dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE) + 1, axis=0) # input shape to the model has to reflect Dense layer input requirements (ndim=2)
print()
print("# Create dummy inputs (have to be same size as input_size)")
print(dummy_inputs)

# Pass dummy inputs to dummy NBeatsBlock layer
backcast, forecast = dummy_nbeats_block_layer(dummy_inputs)
# These are the activation outputs of the theta layer (they'll be random due to no training of the model)
print()
print(f"Backcast: {tf.squeeze(backcast.numpy())}")
print(f"Forecast: {tf.squeeze(forecast.numpy())}")

print()
print("# Create NBEATS data inputs (NBEATS works with univariate time series)")
print(bitcoin_prices.head())

print()
print("# Add windowed columns")
print(bitcoin_prices_nbeats.dropna().head())

print()
print("# Make tensors")
tensor_1 = tf.range(10) + 10
tensor_2 = tf.range(10)

print("# Subtract")
subtracted = layers.subtract([tensor_1, tensor_2])

print("# Add")
added = layers.add([tensor_1, tensor_2])

print(f"Input tensors: {tensor_1.numpy()} & {tensor_2.numpy()}")
print(f"Subtracted: {subtracted.numpy()}")
print(f"Added: {added.numpy()}")

