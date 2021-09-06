import collections
import os
import json
import numpy as np
import time
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def create_image_path_to_caps(annotation_file, image_folder, ds_flag):
  with open(annotation_file, 'r') as f:
    annotations = json.load(f)

  image_path_to_caption = collections.defaultdict(list)
  for val in annotations['annotations']:
    caption = "<s> {} s>".format(val['caption'])
    image_path = image_folder + ds_flag + '%012d.jpg' % (val['image_id'])
    image_path_to_caption[image_path].append(caption)

  return image_path_to_caption

# Process image
def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (299, 299))
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path

def create_image_model():
  image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                  weights='imagenet')
  new_input = image_model.input
  hidden_layer = image_model.layers[-1].output
  return tf.keras.Model(new_input, hidden_layer)

def prepare_image_features(image_keys, image_features_extract_model):
  image_dataset = tf.data.Dataset.from_tensor_slices(image_keys)
  image_dataset = image_dataset.map(load_image,
                                    num_parallel_calls=AUTOTUNE).batch(16)

  print("image path ds {} image_keys {}".format(image_dataset, len(image_keys)))

  t = time.time()
  cnt = 0
  for img, path in image_dataset:
    batch_features = image_features_extract_model(img) # [batch_size, width, height, dim]
    batch_features = tf.reshape(batch_features,
                                [batch_features.shape[0], -1, batch_features.shape[3]])
    for bf, p in zip(batch_features, path):
      path_of_feature = p.numpy().decode("utf-8")
      print("save image {} data shape {} zeros {} {}".format(path_of_feature, np.shape(bf.numpy()), np.count_nonzero(bf.numpy()), np.count_nonzero(bf.numpy()==0)))
      np.save(path_of_feature, bf.numpy())
      cnt+=1

def fit_tokenizer(captions, **kwargs):
  top_k = kwargs['top_k']
  tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,'
                                                              '-/:;=?@[\]^_`{'
                                                              '|}~')
  tokenizer.fit_on_texts(captions)
  tokenizer.word_index['<pad>'] = 0
  tokenizer.index_word[0] = '<pad>'
  return tokenizer

def create_text_id_sequence(texts, tokenizer):
  text_seqs = tokenizer.texts_to_sequences(texts)
  text_vec = tf.keras.preprocessing.sequence.pad_sequences(text_seqs,
                                                           padding='post')
  return text_seqs, text_vec

# Process captions
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def align_image_cap_vec(image_path_to_caption, image_paths):
  captions = []
  img_name_vector = []

  for image_path in image_paths:
    caption_list = image_path_to_caption[image_path]
    captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))

  return captions, img_name_vector
