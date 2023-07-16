from model8_init import *

model8.load_weights('data/model8.h5')

# Load the saved history object from a file
with open('data/history8.pkl', 'rb') as f:
    history8 = pickle.load(f)

print(model8.summary())
plot_curves(history8, 8)

# Get the class names
print(class_names)

download_resource('https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg')

steak = mpimg.imread("data/03-steak.jpeg")
plt.figure()
plt.imshow(steak)
plt.axis(False)
plt.savefig('data/image.png', format='png')

# Check the shape of our image
print(f"steak.shape: {steak.shape}")

# Load in and preprocess our custom image
steak = load_and_prep_image("data/03-steak.jpeg")
print(f"steak: {steak}")

# Add an extra axis
print(f"Shape before new dimension: {steak.shape}")
steak = tf.expand_dims(steak, axis=0) # add an extra dimension at axis 0
#steak = steak[tf.newaxis, ...] # alternative to the above, '...' is short for 'every other dimension'
print(f"Shape after new dimension: {steak.shape}")
print(f"steak: {steak}")

# Make a prediction on custom image tensor
pred = model8.predict(steak)
print(f"pred: {pred}")

# We can index the predicted class by rounding the prediction probability
pred_class = class_names[int(tf.round(pred)[0][0])]
print(f"pred_class: {pred_class}")

