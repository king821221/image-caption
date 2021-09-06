import collections
import os
import json
import numpy as np
import time
import tensorflow as tf

from data_utils import create_image_path_to_caps
from data_utils import create_image_model 
from data_utils import prepare_image_features 

FILE_PATH='/node4/jianwang/models/official/image_caption'

# TRAIN+EVAL
# Download caption annotation files
annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath(FILE_PATH) + annotation_folder):
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                           cache_subdir=os.path.abspath(FILE_PATH),
                                           origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                           extract=True)
  os.remove(annotation_zip)

# Download image files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath(FILE_PATH) + image_folder):
  image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath(FILE_PATH),
                                      origin='http://images.cocodataset.org/zips/train2014.zip',
                                      extract=True)
  os.remove(image_zip)

# TEST
# Download image files
image_folder = '/val2014/'
if not os.path.exists(os.path.abspath(FILE_PATH) + image_folder):
  image_zip = tf.keras.utils.get_file('val2014.zip',
                                      cache_subdir=os.path.abspath(FILE_PATH),
                                      origin='http://images.cocodataset.org/zips/val2014.zip',
                                      extract=True)
  os.remove(image_zip)

annotation_file_train = os.path.abspath(FILE_PATH) + '/annotations/captions_train2014.json'
image_folder_train = os.path.abspath(FILE_PATH) + '/train2014/'

annotation_file_test = os.path.abspath(FILE_PATH) + '/annotations/captions_val2014.json'
image_folder_test = os.path.abspath(FILE_PATH) + '/val2014/'

print(annotation_file_train)
print(image_folder_train)
print(annotation_file_test)
print(image_folder_test)

ds_flag = 'COCO_train2014_'
image_path_to_caption_train = create_image_path_to_caps(
    annotation_file_train,
    image_folder_train,
    ds_flag)

ds_flag = 'COCO_val2014_'
image_path_to_caption_test = create_image_path_to_caps(
    annotation_file_test,
    image_folder_test,
    ds_flag)

img_ext_model = create_image_model()
prepare_image_features(list(image_path_to_caption_train.keys()),
                       img_ext_model)
prepare_image_features(list(image_path_to_caption_test.keys()),
                       img_ext_model)
