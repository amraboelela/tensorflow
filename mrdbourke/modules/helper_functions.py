### We create a bunch of helpful functions throughout the course.
### Storing them here so they're easily accessible.

import os
from os import path

from sklearn.compose import make_column_transformer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import make_circles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras import layers, Sequential, mixed_precision
from tensorflow.keras.activations import linear, relu, sigmoid
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Activation, Bidirectional, Conv1D, Conv2D, Dense, Embedding, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPool1D, GlobalMaxPool2D, GRU, LSTM, MaxPool2D, RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth, TextVectorization
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.losses import binary_crossentropy, BinaryCrossentropy, mae, SparseCategoricalCrossentropy
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import tensorflow_datasets as tfds

import tensorflow_hub as hub

import datetime
import itertools
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import pickle
import random
import subprocess

IMAGE_SHAPE = (224, 224)

subprocess.run(['mkdir', '-p', 'data/images'])
    
# NEW: Newer versions of TensorFlow (2.10+) can use the tensorflow.keras.layers API directly for data augmentation
data_augmentation = Sequential( [
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomHeight(0.2),
        RandomWidth(0.2),
        # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
    ],
    name = "data_augmentation"
)

def mean_absolute_error(y_test, y_pred):
    """
    Calculuates mean absolute error between y_test and y_preds.
    """
    return tf.metrics.mean_absolute_error(y_test,
                                        y_pred)
  
def mean_squared_error(y_test, y_pred):
    """
    Calculates mean squared error between y_test and y_preds.
    """
    return tf.metrics.mean_squared_error(y_test,
                                       y_pred)
                                       
# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filePath, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filePath)
    # Decode it into a tensor
    img = tf.io.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img/255.
    else:
        return img

# Note: The following confusion matrix code is a remix of Scikit-Learn's
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
  
def view_random_image(target_dir, target_class):
    # Setup target directory (we'll view images from here)
    target_folder = target_dir+target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off");
    print(f"Image shape: {img.shape}") # show the shape of the image
    plt.savefig('data/images/random_image.png', format='png')
    return img
  
# Create a function for plotting a random image along with its prediction
def plot_random_image(model, images, true_labels, classes):
    """Picks a random image, plots it and labels it with a predicted and truth label.

    Args:
    model: a trained model (trained on data similar to what's in images).
    images: a set of random images (in tensor form).
    true_labels: array of ground truth labels for images.
    classes: array of class names for images.

    Returns:
    A plot of a random image from `images` with a predicted class label from `model`
    as well as the truth class label from `true_labels`.
    """
    # Setup random integer
    i = random.randint(0, len(images))

    # Create predictions and targets
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28)) # have to reshape to get into right size for model
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    plt.figure()
    # Plot the target image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Change the color of the titles depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    # Add xlabel information (prediction/true label)
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                   100*tf.reduce_max(pred_probs),
                                                   true_label),
             color=color) # set the color to green or red
    plt.savefig('data/images/random_image_predict.png', format='png')
    
# Make a function to predict on images and plot them (works with multi-class)
def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1: # check for multi-class
        pred_class = class_names[pred.argmax()] # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

    plt.figure()
    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
      
    filenameTokens = filename.split(".")
    imageName = filenameTokens[0]
    plt.savefig('data/images/' + imageName + '.png', format='png')
  
def plot_decision_boundary(model, X, y, index):
    """
    Plots the decision boundary created by a model predicting on X.
    This function has been adapted from two phenomenal resources:
    1. CS231n - https://cs231n.github.io/neural-networks-case-study/
    2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if model.output_shape[-1] > 1: # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)
  
    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig('data/images/decision_boundary' + str(index) + '.png', format='png')

def tensorboard_callback(dir_name):
    """
    Creates a TensorBoard callback instand to store log files.

    Stores log files with the filepath:
    "dir_name/current_datetime/"

    Args:
    dir_name: target directory to store TensorBoard log files
    """
    log_dir = "data/" + dir_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result = TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return result

def checkpoint_path(index):
    return "data/model" + str(index) + "/checkpoint.ckpt"
    
def checkpoint_callback(index):
    result = ModelCheckpoint(
        filepath=checkpoint_path(index),
        save_weights_only=True, # set to False to save the entire model
        save_best_only=True, # save only the best model weights instead of a model every epoch
        save_freq="epoch", # save every epoch
        verbose=1
    )
    return result

def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions,
    index
):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend()
    plt.savefig('data/images/predictions' + str(index) + '.png', format='png')

def create_model(model_url, num_classes=10):
    """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in output layer,
      should be equal to number of target classes, default 10.

    Returns:
    An uncompiled Keras Sequential model with model_url as feature
    extractor layer and Dense output layer with num_classes outputs.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(
        model_url,
        trainable=False, # freeze the underlying patterns
        name='feature_extraction_layer',
        input_shape=IMAGE_SHAPE+(3,)
    ) # define the input image shape

    # Create our own model
    model = Sequential([
        feature_extractor_layer, # use the feature extraction layer as the base
        Dense(num_classes, activation='softmax', name='output_layer') # create our own output layer
    ])
    return model
  
# Plot the validation and training data separately
def plot_curves(history, index):
    print("")
    #plot_loss_curves(history, index)
    plot_accuracy_curves(history, index)

def plot_loss_curves(history, index):
    """
    Returns separate loss curves for training and validation metrics.

    Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """
    loss = history['loss']
    print("loss: " + str(loss))
    val_loss = history['val_loss']

    epochs = range(len(history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('data/images/loss' + str(index) + '.png', format='png')

def plot_accuracy_curves(history, index):
    """
    Returns separate accuracy curves for training and validation metrics.

    Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """

    accuracy = history['accuracy']
    print("accuracy: " + str(accuracy))
    val_accuracy = history['val_accuracy']

    epochs = range(len(history['loss']))

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('data/images/accuracy' + str(index) + '.png', format='png')

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))
  
def relu(x):
    return tf.maximum(0, x)
  
def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here)
    """
    
    # Get original history measurements
    acc = original_history["accuracy"]
    loss = original_history["loss"]

    val_acc = original_history["val_accuracy"]
    val_loss = original_history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history["accuracy"]
    total_loss = loss + new_history["loss"]

    total_val_acc = val_acc + new_history["val_accuracy"]
    total_val_loss = val_loss + new_history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('data/images/historys.png', format='png')

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.

    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
  
def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1
    }
    return model_results

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=True):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).

    Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
    """
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes), # create enough axis slots for each class
        yticks=np.arange(n_classes),
        xticklabels=labels, # axes will labeled with class names (if they exist) or ints
        yticklabels=labels
    )

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                  horizontalalignment="center",
                  color="white" if cm[i, j] > threshold else "black",
                  size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                  horizontalalignment="center",
                  color="white" if cm[i, j] > threshold else "black",
                  size=text_size)

    # Save the figure to the current working directory
    fig.savefig("data/images/confusion_matrix.png")

def download(url):
    urlTokens = url.split("/")
    resourceFile = urlTokens[-1]
    resourceTokens = resourceFile.split(".")
    resource = resourceTokens[0]
    resourceExtension = resourceTokens[-1]
    os.chdir("data")
    if resourceExtension == "zip":
        if not path.exists(resource) and not path.exists(resourceFile):
            subprocess.run(['wget', url])
            subprocess.run(['unzip', resourceFile])
            if path.exists(resource):
                subprocess.run(['rm', resourceFile])
    elif resourceExtension == "jpeg":
        os.chdir("images")
        if not path.exists(resourceFile):
            subprocess.run(['wget', url])
        os.chdir("..")
    else:
        if not path.exists(resourceFile):
            subprocess.run(['wget', url])
    os.chdir("..")

def find_subdirectories_with_leaf(root_dir, leaf_dir):
    """
    Find all subdirectory paths containing the specific leaf directory.

    Parameters:
    root_dir (str): The root directory to start the search.
    leaf_dir (str): The name of the leaf directory you want to find.

    Returns:
    list: A list of subdirectory paths that contain the leaf directory.
    """
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if leaf_dir in dirnames:
            matching_dirs.append(os.path.join(dirpath, leaf_dir))
    return matching_dirs

def remove_subdirectories_with_leaf(root_dir, leaf_dir):
    subdirectories_with_train = find_subdirectories_with_leaf(root_dir, leaf_dir)
    for train_path in subdirectories_with_train:
        subprocess.run(['rm', '-r', train_path])
    
def save_tensor(tensor, name):
    # Convert the tensor to a string
    tensor_string = tf.io.serialize_tensor(tensor)
    # Specify the file path to save the tensor
    file_path = "data/" + name + ".tfd"
    # Write the tensor string to the file
    tf.io.write_file(file_path, tensor_string)

def read_tensor(name):
    # Specify the file path of the saved tensor
    file_path = "data/" + name + ".tfd"
    try:
        # Read the tensor string from the file
        tensor_string = tf.io.read_file(file_path)
    except Exception as e:
        return None
    # Deserialize the tensor string to a tensor
    tensor = tf.io.parse_tensor(tensor_string, out_type=tf.float32)
    return tensor
    
# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
    return tf.cast(image, tf.float32), label # return (float32_image, label) tuple

# Create a helper function to compare our baseline results to new model results
def compare_baseline_to_new_results(baseline_results, new_model_results):
    for key, value in baseline_results.items():
        print(f"Baseline {key}: {value:.2f}, New {key}: {new_model_results[key]:.2f}, Difference: {new_model_results[key]-value:.2f}")
