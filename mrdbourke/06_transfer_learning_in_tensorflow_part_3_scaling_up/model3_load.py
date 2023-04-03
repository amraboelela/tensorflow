from common import *

model_name = '06_101_food_class_10_percent_saved_big_dog_model'

if not path.exists('data/' + model_name):
    # Download pre-trained model from Google Storage (like a cooking show, I trained this model earlier, so the results may be different than above)
    subprocess.run(['wget', 'https://storage.googleapis.com/ztm_tf_course/food_vision/' + model_name  + '.zip'])
    unzip_data(model_name + '.zip')
    subprocess.run(['rm', model_name + '.zip'])
    subprocess.run(['mv', model_name, 'data'])

# Note: loading a model will output a lot of 'WARNINGS', these can be ignored: https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training
# There's also a thread on GitHub trying to fix these warnings: https://github.com/tensorflow/tensorflow/issues/40166
# model = tf.keras.models.load_model("drive/My Drive/tensorflow_course/101_food_class_10_percent_saved_big_dog_model/") # path to drive model
model = tf.keras.models.load_model('data/' + model_name) # don't include ".zip" in loaded model path
