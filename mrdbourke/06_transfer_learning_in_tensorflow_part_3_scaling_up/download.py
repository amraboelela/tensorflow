from os import path
import subprocess
from helper_functions import *

def download_resource(resource):
    if not path.exists('data/' + resource):
        subprocess.run(['rm', resource + '.zip'])
        subprocess.run(['wget', 'https://storage.googleapis.com/ztm_tf_course/food_vision/' + resource + '.zip'])
        subprocess.run(['unzip', resource])
        subprocess.run(['mv', resource, 'data'])
        subprocess.run(['rm', resource + '.zip'])
        subprocess.run(['rm', '-r', '__MACOSX'])
    
subprocess.run(['mkdir', '-p', 'data'])
download_resource('101_food_classes_10_percent')
walk_through_dir("data/101_food_classes_10_percent")

download_resource('custom_food_images')

