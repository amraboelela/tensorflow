from model3_init import *

model3.load_weights('data/model3.keras')

# Load the saved history object from a file
with open('data/history3.pkl', 'rb') as f:
    history3 = pickle.load(f)

print(model3.summary())
plot_curves(history3, 3)

download('https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-pizza-dad.jpeg')
download('https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg')
download('https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-hamburger.jpeg')
download('https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-sushi.jpeg')

print("# Make a prediction using model3")
pred_and_plot(model=model3,
              filename="03-steak.jpeg",
              class_names=class_names)
pred_and_plot(model3, "03-sushi.jpeg", class_names)
pred_and_plot(model3, "03-pizza-dad.jpeg", class_names)

# Load in and preprocess our custom image
img = load_and_prep_image("03-steak.jpeg")

# Make a prediction
pred = model3.predict(tf.expand_dims(img, axis=0))
plt.figure()
# Match the prediction class to the highest prediction probability
pred_class = class_names[pred.argmax()]
plt.imshow(img)
plt.title(pred_class)
plt.axis(False)
plt.savefig('data/images/03-steak-2.png', format='png')

pred_and_plot(model3, "03-hamburger.jpeg", class_names)

# Save a model
model3.save("data/model3-full.keras")

# Load in a model and evaluate it
loaded_model3 = load_model("data/model3-full.keras")
loaded_model3.evaluate(test_data)

