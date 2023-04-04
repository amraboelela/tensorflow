from model3_evaluate_load import *
from model3_load import *

download_resource('custom_food_images')

# Get custom food images filepaths
custom_food_images = ["data/custom_food_images/" + img_path for img_path in os.listdir("data/custom_food_images")]
print(custom_food_images)

# Make predictions on custom food images
plt.figure(figsize=(15, 10))
count = 1
for img in custom_food_images:
  plt.subplot(3, 3, count)
  count += 1
  img = load_and_prep_image(img, scale=False) # load in target image and turn it into tensor
  pred_prob = model.predict(tf.expand_dims(img, axis=0)) # make prediction on image with shape [None, 224, 224, 3]
  pred_class = class_names[pred_prob.argmax()] # find the predicted class label
  # Plot the image with appropriate annotations
  plt.figure()
  plt.imshow(img/255.) # imshow() requires float inputs to be normalized
  plt.title(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")
  plt.axis(False)
plt.savefig('plot.png', format='png')
subprocess.run(['mv', 'plot.png', imagePath + "/plot3.png"])

