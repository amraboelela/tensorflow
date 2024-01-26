from common import *

print()
print("# Parse dates and set date column to index")

print(df.head())

print(df.info())

print()
print("# How many samples do we have?")
print(len(df))

print()
print("# Only want closing price for each day")
print(bitcoin_prices.head())

bitcoin_prices.plot(figsize=(10, 7))
plt.ylabel("BTC Price")
plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
plt.legend(fontsize=14)
plt.savefig('data/images/Price_of_Bitcoin.png', format='png')

print()
print("# View first 10 of each")
print(timesteps[:10], btc_price[:10])

plt.figure(figsize=(10, 7))
plt.plot(timesteps, btc_price)
plt.title("Price of Bitcoin from 1 Oct 2013 to 18 May 2021", fontsize=16)
plt.xlabel("Date")
plt.ylabel("BTC Price");
plt.savefig('data/images/BTC_Price.png', format='png')

print()
print("# Get bitcoin date array")
print(timesteps[:10], prices[:10])

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print()
print("# Wrong way to make train/test sets for time series")
X_train, X_test, y_train, y_test = train_test_split(
    timesteps, # dates
    prices, # prices
    test_size=0.2,
    random_state=42)
print("# Let's plot wrong train and test splits")
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, s=5, label="Train data")
plt.scatter(X_test, y_test, s=5, label="Test data")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.legend(fontsize=14)
plt.savefig('data/images/train_test_splits_wrong.png', format='png')

print()
print("# Create train data splits (everything before the split)")
X_train, y_train = timesteps[:split_size], prices[:split_size]

print("# Create test data splits (everything after the split)")
X_test, y_test = timesteps[split_size:], prices[split_size:]

print(len(X_train), len(X_test), len(y_train), len(y_test))

print()
print("# Plot correctly made splits")
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, s=5, label="Train data")
plt.scatter(X_test, y_test, s=5, label="Test data")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.legend(fontsize=14)
plt.savefig('data/images/train_test_splits.png', format='png')

print()
print("# Try out our plotting function")
plt.figure(figsize=(10, 7))
plot_time_series(timesteps=X_train, values=y_train, label="Train data", name="time_series")
plot_time_series(timesteps=X_test, values=y_test, label="Test data", name="time_series")

print()
print("# Test out the window labelling function")
test_window, test_label = get_labelled_windows(tf.expand_dims(tf.range(8)+1, axis=0), horizon=HORIZON)
print(f"Window: {tf.squeeze(test_window).numpy()} -> Label: {tf.squeeze(test_label).numpy()}")

print(len(full_windows), len(full_labels))

print()
print("# View the first 3 windows/labels")
for i in range(3):
    print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")
  
print()
print("# View the last 3 windows/labels")
for i in range(3):
    print(f"Window: {full_windows[i-3]} -> Label: {full_labels[i-3]}")

print()
print("# Find average price of Bitcoin in test dataset")
print(tf.reduce_mean(y_test).numpy())

print(len(train_windows), len(test_windows), len(train_labels), len(test_labels))

print(train_windows[:5], train_labels[:5])

print()
print("# Check to see if same (accounting for horizon and window size)")
print(np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), y_train[WINDOW_SIZE:]))
