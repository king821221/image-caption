import sys
import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# your model focuses on during captioning
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import collections
import random
import logging
import nltk
import numpy as np
import os
import time
from PIL import Image
from scipy import stats
from data.data_util import create_image_path_to_caps
from data.data_util import create_image_model
from data.data_util import align_image_cap_vec
from data.data_util import fit_tokenizer
from data.data_util import calc_max_length
from data.data_util import create_text_id_sequence

from encoder.mlp_encoder import MlpEncoder
from decoder.embedding import EmbeddingLayer 
from decoder.recurrent_decoder import RecurrentDecoder
from decoder.lstm_recurrent_decoder import LstmDecoder 

from beam_search.beam_search import  create_initial_beam_state_batch
from beam_search.beam_search import beam_search_step_batch
from beam_search.beam_search import BeamSearchConfig
from beam_search.beam_search import choose_top_k
from beam_search.beam_search import loc_optimal_beam_path_batch

from scheduled_sample.scheduled_sampling import ScheduledSampling
from scheduled_sample.scheduled_sampling import PredictProbConfidenceFunc
from scheduled_sample.scheduled_sampling import ConfidenceScheduledSampling
from scheduled_sample.scheduled_sampling import InverseSigmoidDecay

from utils.train_utils import train_step_bso
from utils.train_utils import train_step_reinforce
from utils.train_utils import train_step_self_critic
from utils.train_utils import train_step_actor_critic 
from utils.tf_utils import tf_summary
from utils.actor_critic import TargetSequenceEncoder 
from utils.reinforce import SequenceBleuRewardFn
from utils.reinforce import SequencePerStepRewardFn 
from utils.reinforce import RnnDecoderNetwork 

AUTOTUNE = tf.data.experimental.AUTOTUNE

FILE_PATH = '.'

top_k = 5000

SCHED_SAMPLE_INVERSE_DECAY_K = 8

beam_width = 5

TRAIN_SIZE =200
TEST_SIZE =11

annotation_file_train = os.path.join(FILE_PATH, 'annotations/captions_train2014.json')
image_folder_train = os.path.join(FILE_PATH, 'train2014/')

annotation_file_test = os.path.join(FILE_PATH, 'annotations/captions_val2014.json')
image_folder_test = os.path.join(FILE_PATH, 'val2014/')

ds_flag = 'COCO_train2014_'
image_path_to_caption_train = create_image_path_to_caps(annotation_file_train,
                                                        image_folder_train,
                                                        ds_flag)

print("# imgs TRAIN {}".format(len(image_path_to_caption_train)))
cs = [len(c) for c in list(image_path_to_caption_train.values())]
print("# captions per img TRAIN {}".format(stats.describe(cs)))

c = 0
for img_path_key, captions in image_path_to_caption_train.items():
  if len(captions) > 1:
    print("sample TRAIN image with multiple caps {}, caps {}"
          .format(img_path_key, captions))
    c+=1
    if c > 3:
      break

ds_flag = 'COCO_val2014_'
image_path_to_caption_test = create_image_path_to_caps(
    annotation_file_test,
    image_folder_test,
    ds_flag)

print("# imgs TEST {}".format(len(image_path_to_caption_test)))
cs = [len(c) for c in list(image_path_to_caption_test.values())]
print("# captions per img TEST {}".format(stats.describe(cs)))

c = 0
for img_path_key, captions in image_path_to_caption_test.items():
  if len(captions) > 1:
    print("sample TEST image with multiple caps {}, caps {}"
          .format(img_path_key, captions))
    c+=1
    if c > 3:
      break

train_image_paths = list(image_path_to_caption_train.keys())
random.shuffle(train_image_paths)
if TRAIN_SIZE > 0:
  train_image_paths = train_image_paths[0:TRAIN_SIZE]

train_captions, train_img_name_vec = align_image_cap_vec(
    image_path_to_caption_train,
    train_image_paths)

for idx in range(0, 10):
    tr_cap = train_captions[idx]
    tr_img_name = train_img_name_vec[idx]
    print("train image {} mapping caption {}".format(tr_img_name, tr_cap))

test_image_paths = list(image_path_to_caption_test.keys())
if TEST_SIZE > 0:
  test_image_paths = test_image_paths[0:TEST_SIZE]
test_captions, test_img_name_vec = align_image_cap_vec(
    image_path_to_caption_test,
    test_image_paths)

for idx in range(0, 10):
    ts_cap = test_captions[idx]
    ts_img_name = test_img_name_vec[idx]
    print("test image {} mapping caption {}".format(ts_img_name, ts_cap))

tokenizer = fit_tokenizer(train_captions + test_captions, top_k=top_k)

print("tokenizer {}".format(tokenizer.get_config()))

for i in range(0,20):
  print("word index {} word {}".format(i, tokenizer.index_word[i]))

def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8') + '.npy')
  return img_tensor, cap

BATCH_SIZE = 64
BATCH_SIZE = 32

EVAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
embedding_dim = 256
units = 512
vocab_size = top_k + 1

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape

def create_train_eval_dataset(train_img_name_vec,
                              train_captions,
                              buffer_size=1000,
                              batch_size=64,
                              eval_batch_size=64):
  train_seqs, train_cap_vector = create_text_id_sequence(train_captions,
                                                         tokenizer)

  print("# of train cap vec {}".format(len(train_cap_vector)))
  for idx in range(10):
    print("Inspect train sequence padded caption at idx {}".format(idx))
    print(train_cap_vector[idx])
    print(train_seqs[idx])

  max_length = calc_max_length(train_seqs)
  print("max_length of train_seq {}".format(max_length))

  img_to_cap_vector = collections.defaultdict(list)

  for img, cap in zip(train_img_name_vec, train_cap_vector):
    img_to_cap_vector[img].append(cap)

  img_keys = list(img_to_cap_vector.keys())
  random.shuffle(img_keys)

  slice_index = int(len(img_keys) * 4/5)

  img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

  img_name_train = []
  cap_train = []
  for imgt in img_name_train_keys:
    cap_train.extend(img_to_cap_vector[imgt])
    img_name_train.extend([imgt] * len(img_to_cap_vector[imgt]))

  print("img_name_train {} cap_train {}".format(
      len(img_name_train), len(cap_train)))

  assert len(cap_train) == len(img_name_train)

  img_name_val = []
  cap_val = []
  for imgv in img_name_val_keys:
    cap_val.extend(img_to_cap_vector[imgv])
    img_name_val.extend([imgv] * len(img_to_cap_vector[imgv]))

  print("img_name_val {} cap_val {}".format(len(img_name_val), len(cap_val)))

  for i in range(3):
    print("train img sample {} caption {}".format(img_name_train[i], cap_train[i]))
  for i in range(3):
    print("eval img sample {} caption {}".format(img_name_val[i], cap_val[i]))

  # TRAIN dataset
  dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
  print("image cap train ds: {}".format(dataset))
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1,item2], [tf.float32, tf.int32]))
  print("image cap train mapped ds: {}".format(dataset))
  dataset = dataset.shuffle(buffer_size).batch(batch_size)
  train_dataset = dataset.prefetch(buffer_size=AUTOTUNE)

  # DEV dataset
  dataset = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
  print("image cap eval ds: {}".format(dataset))
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1,item2], [tf.float32, tf.int32]))
  print("image cap eval mapped ds: {}".format(dataset))
  dataset = dataset.batch(eval_batch_size)
  dev_dataset = dataset.prefetch(buffer_size=AUTOTUNE)

  return train_dataset, dev_dataset, max_length

def create_test_dataset(test_img_name_vec, test_captions, test_batch_size=64):
  test_seqs, test_cap_vector = create_text_id_sequence(test_captions,
                                                       tokenizer)

  print("# of test cap vec {}".format(len(test_cap_vector)))
  for idx in range(10):
    print("Inspect test sequence padded caption at idx {}".format(idx))
    print(test_cap_vector[idx])
    print(test_seqs[idx])

  max_length = calc_max_length(test_seqs)
  print("max_length of test_seq {}".format(max_length))

  img_to_cap_vector = collections.defaultdict(list)

  for img, cap in zip(test_img_name_vec, test_cap_vector):
    img_to_cap_vector[img].append(cap)

  img_name_test = []
  cap_test = []
  for imgt in img_to_cap_vector.keys():
    cap_test.extend(img_to_cap_vector[imgt])
    img_name_test.extend([imgt] * len(img_to_cap_vector[imgt]))

  assert len(cap_test) == len(img_name_test)

  print("img_name_test {} cap_test {}".format(len(img_name_test), len(cap_test)))

  for i in range(3):
    print("test img sample {} caption {}".format(img_name_test[i],
                                                 cap_test[i]))

  dataset = tf.data.Dataset.from_tensor_slices((img_name_test, cap_test))
  print("image cap test ds: {}".format(dataset))
  dataset = dataset.map(
        lambda item1, item2: tf.numpy_function(map_func, [item1, item2],
                                               [tf.float32, tf.int32]))
  print("image cap test mapped ds: {}".format(dataset))
  dataset = dataset.batch(test_batch_size)
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  return dataset, max_length


# Embedding layer to encode tokens to token embeddings
embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)

# Source feature encoder
encoder = MlpEncoder(embedding_dim, name='mlp_feature_encoder')

def embedding_lookup(inputs, **kwargs):
  return embedding_layer(inputs)

# Target sequence decoder
# As actor in Actor-Critic RL training
decoder = LstmDecoder(embedding_lookup,
                      embedding_dim,
                      units,
                      vocab_size,
                      reset_state_from_feat=1,
                      name='target_sequence_decoder')
decoder.create_variables()
# Optimizer of source feature encoder and target sequence decoder
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

##############################
# For actor-critic RL training
##############################
# Optimizer of target sequence encoder and critic decoder 
critic_optimizer = tf.keras.optimizers.Adam()
# Target sequence decoder as Critic in training
# Critic is ONLY used in training, not in inference
critic_decoder = LstmDecoder(embedding_lookup,
                             embedding_dim,
                             units,
                             vocab_size,
                             reset_state_from_feat=1,
                             name='target_sequence_critic_decoder')
critic_decoder.create_variables()
target_sequence_encoder = TargetSequenceEncoder(embedding_lookup,
                                                units,
                                                embedding_dim,
                                                0)
target_sequence_encoder.create_variables()
 
delayed_actor_decoder = decoder.copy()
delayed_actor_decoder.create_variables()
init_delayed_actor = True
init_target_critic = True

target_critic_decoder = critic_decoder.copy()
target_critic_decoder.create_variables()

##############################
# End prepare for actor-critic RL training
##############################

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)

  logging.debug("loss_function loss {} mask {}".format(
     loss_, mask
  ))

  loss_ *= mask

  return loss_, mask 

checkpoint_path = "./checkpoints/train"
checkpoint_path = "./checkpoints_v3_beam_search/train"
checkpoint_path = "./checkpoints_v4_full/train"
checkpoint_path = "./checkpoints_v4_conf_sched_sample_beam_search/train"
checkpoint_path = "./checkpoints_v4_sched_sample_beam_search/train"
checkpoint_path = "./checkpoints_v5_bso/train"
checkpoint_path = "./checkpoints_v6_reinforce/train"
checkpoint_path = "./checkpoints_v2_transformer/train"
checkpoint_path = "./checkpoints_v7_self_critic/train"
checkpoint_path = "./checkpoints_v7_actor_critic/train"
summary_dir = 'tf_summary'
summary_dir = 'tf_summary_full'
summary_dir = 'tf_summary_conf_sched_sample'
summary_dir = 'tf_summary_sched_sample'
summary_dir = 'tf_summary_bso'
summary_dir = 'tf_summary_self_critic'
summary_dir = 'tf_summary_reinforce'
summary_dir = 'tf_summary_transformer'
summary_dir = 'tf_summary_actor_critic'
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           embedding_layer=embedding_layer,
                           critic_decoder=critic_decoder,
                           target_sequence_encoder=target_sequence_encoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)

loss_plot = []

writer = tf.summary.create_file_writer(summary_dir)

# Model
## training

#@tf.function
def train_step(img_tensor, target, step, **kwargs):
  sequence_loss = 0.0
  token_loss = 0.0
  token_mask = 0

  dec_input_fn = kwargs.get('dec_input_fn')

  # dec_input: [batch_size, 1]
  dec_input = tf.expand_dims([tokenizer.word_index["<s>"]] * target.shape[0],
                             axis=1)

  with tf.GradientTape() as tape:

    # features: [batch_size, width * height, embedding_dim]
    features = encoder(img_tensor, training=True)
    tf_summary('decode_train_features', features, mode='', step=step)

    # hidden: [batch_size, units]
    hidden = decoder.reset_state(target, {"features": features})

    attention_weights_sum = 0.0
    output_attention_weights = False

    for i in range(1, target.shape[1]):
      predictions, hidden, attention_weights, context_vec = decoder((dec_input,
                                                                     {"features": features},
                                                                     hidden),
                                                                    training=True)

      logging.debug("train sequence idx {} dec_input {} hidden {}".format(i, dec_input, hidden))

      tf_summary('decode_train_predictions', predictions, step=step, mode='histogram')
      if attention_weights is not None:
        tf_summary('decode_train_attention_weights', attention_weights, step=step, mode='histogram')
      for hidx, h_state in enumerate(hidden):
        tf_summary('decode_train_hidden_states_{}'.format(hidx), h_state, step=step, mode='histogram')
      if context_vec is not None:
        tf_summary('decode_train_context_vec', context_vec, step=step, mode='histogram')

      logging.debug("train step {} predictions {}".format(i, (predictions.shape)))
      for hidx, h_state in enumerate(hidden):
        logging.debug("train step {} hidden state {} {}".format(i, hidx, h_state))
      if attention_weights is not None:
        logging.debug("train step {} attention_weights {}".format(i, (attention_weights.shape)))
      if context_vec is not None:
        logging.debug("train step {} context_vec {}".format(i, (context_vec.shape)))

      pt_loss, pt_mask = loss_function(target[:, i], predictions)

      if attention_weights is not None:
        attention_weights = tf.squeeze(attention_weights, -1) 
        attention_weights_sum += attention_weights 
        output_attention_weights = True

      sequence_loss += tf.reduce_mean(pt_loss)
      token_loss += tf.reduce_sum(pt_loss)
      token_mask += tf.reduce_sum(pt_mask)
      dec_input = tf.expand_dims(target[:, i], 1)
      if dec_input_fn:
        dec_from_tgt = dec_input
        dec_input = dec_input_fn(target, predictions, tf.constant(i), tf.constant(step, dtype=tf.int64), **kwargs)
        dec_use_other = tf.logical_not(tf.equal(dec_input, dec_from_tgt))
        dec_use_other = tf.cast(dec_use_other, tf.int32)
        dec_use_other = tf.reduce_mean(dec_use_other)
        tf_summary('decode_train_use_other_tgt', dec_use_other, step=step, mode='')
        tf_summary('decode_train_use_other_tgt_scalar', dec_use_other, step=step, mode='scalar')
        
    logging.debug("train sequence loss {}".format(sequence_loss))
    tf_summary('train_sequence_loss', sequence_loss, step=step, mode='scalar')

    if output_attention_weights:
       atten_w_loss = tf.reduce_sum(tf.math.pow(1.0 - attention_weights_sum, 2), 1)
    else:
       atten_w_loss = tf.zeros((target.shape[0]), dtype=features.dtype)
    lambda_c = 1.0
    atten_w_loss = tf.reduce_mean(atten_w_loss) * lambda_c
    logging.debug("train attention loss {}".format(atten_w_loss))
    tf_summary('train_atten_w_loss', atten_w_loss, step=step, mode='scalar')

    per_token_loss = tf.cond(tf.greater(token_mask, 0), lambda : token_loss / token_mask, lambda: 0.0) 
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    logging.debug("train encoder vars {}".format(encoder.trainable_variables))
    logging.debug("train decoder vars {}".format(decoder.trainable_variables))

    loss = sequence_loss + atten_w_loss 

    logging.debug("training sequence loss {}".format(loss))
    tf_summary('train_loss', loss, step=step, mode='scalar')

    gradients = tape.gradient(loss, trainable_variables)
  
    optimizer.apply_gradients(zip(gradients, trainable_variables))
 
  return loss, per_token_loss, trainable_variables, gradients

def predict_image_caption_sequence(img_tensor_val,
                                   beam_width,
                                   max_predict_length,
                                   step,
                                   training):
  batch_size = img_tensor_val.shape[0]

  # features: [batch_size, w*h, embedding dim]
  features = encoder(img_tensor_val, training=training)
  features = tf.tile(features, [beam_width, 1, 1])
  logging.debug("predict image feature {}".format(features.shape))
  tf_summary('predict_img_encoded_features', features, step=step, mode='')

  # [beam_width, 1]
  dec_input = tf.expand_dims([tokenizer.word_index['<s>']] * beam_width, 1)

  # [batch_size*beam_width, 1]
  dec_input = tf.tile(dec_input, [batch_size, 1])

  logging.debug("predict_caption dec_input {}".format(dec_input.shape))

  # hidden: [[batch_size * beam_width, units]]
  dummy_tgt = tf.zeros_like(dec_input)
  hidden = decoder.reset_state(dummy_tgt, {"features": features})

  logging.debug("predict_caption hidden {}".format(hidden))

  bs_state = create_initial_beam_state_batch(batch_size, beam_width)
  bs_config = BeamSearchConfig(beam_width=beam_width,
                               vocab_size=vocab_size,
                               eos_token=tokenizer.word_index['s>'],
                               length_penalty_weight=1.0,
                               choose_successors_fn=choose_top_k)

  #print("predict caption initial beam state {} config {}".format(
     # bs_state, bs_config
  #))

  bs_sequence = []

  for i in range(max_predict_length):
    # predictions: [1 * beam_width, vocab_size]
    # hidden: [1 * beam_width, state dim]
    # attention_weight: [1 * beam_width, feature length]

    # predictions: [batch_size * beam_width, vocab_size]
    # hidden: [[batch_size * beam_width, units]]
    # attention_weights: [batch_size * beam_width, T, T]
    # context_vec: [batch_size * beam_width, units]
    predictions, hidden, attention_weights, context_vec = decoder((dec_input,
                                                                   {"features": features},
                                                                   hidden),
                                                                   training=False)

    logging.debug("predict step {} predictions {} dec_input {} features {} context vec {}".format(
      i, predictions.shape, dec_input.shape, features.shape, context_vec.shape
    ))
    #for h_state in hidden:
      #print("predict step {} h_state {}".format(i, h_state.shape))
    #print("predict step {} predictions {}".format(i, (predictions.shape)))
    #for hidx, h_state in enumerate(hidden):
      #print("predict step {} hidden {} {}".format(i, hidx, h_state.shape))
    #print("predict step {} attention_weights {}".format(i, (attention_weights.shape)))

    predictions = tf.reshape(predictions, [-1, beam_width, predictions.shape[-1]])

    bs_out, bs_state = beam_search_step_batch(i, predictions, bs_state, bs_config)

    #print("predict step {} bs_out {} bs_state {}".format(i, bs_out, bs_state))

    bs_sequence.append((bs_out, bs_state))

    tf_summary('predict_decode_predictions', predictions, step=step, mode='')
    if attention_weights is not None:
      tf_summary('predict_decode_attention_weights', attention_weights, step=step, mode='')
    for hidx, h_state in enumerate(hidden):
      tf_summary('predict_decode_hidden_{}'.format(hidx), h_state, step=step, mode='')
      logging.debug("decode step {} hidden {}: {}".format(i, hidx, h_state))

    # Grow from beam out's predicted id
    # dec_input: [batch_size * beam_width, 1]
    dec_input = tf.reshape(bs_out.predicted_ids, [-1, 1])

  decode_sequence = loc_optimal_beam_path_batch(bs_sequence)

  return decode_sequence

print("start training epoch {}".format(start_epoch))

train_dataset, dev_dataset, max_length = create_train_eval_dataset(
    train_img_name_vec,
    train_captions,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)

step_c = 0
step_e = 0
start_token_id = tokenizer.word_index["<s>"] 
eos_token_id = tokenizer.word_index["s>"] 

EPOCHS = 20
EPOCHS = 8

EVAL_EPOCHS = EPOCHS + (-5)
EVAL_EVERY_EPOCHS = 1

CRITIC_FREEZE_EPOCHS = EPOCHS + (-1)

TRAIN_M = 4

XENT_EPOCHS = EPOCHS + (-100)
XENT_SCHED_EPOCHS = EPOCHS + (-10)

CONFIDENCE_SAMPLE_M = 0

decay_fn = InverseSigmoidDecay(SCHED_SAMPLE_INVERSE_DECAY_K)
sched_sampling = ScheduledSampling(decay_fn)
prob_conf_fn = PredictProbConfidenceFunc()

def gold_prob_thr_fn(step, **kwargs):
  gold_prob = 0.9
  return gold_prob * decay_fn(step)

def rand_prob_thr_fn(step, **kwargs):
  return 0.95

conf_sched_sampling = ConfidenceScheduledSampling(prob_conf_fn, gold_prob_thr_fn, rand_prob_thr_fn)

def beam_width_fn(start_beam_width, end_beam_width, growth, epoch, start_epoch):
  num_epochs = epoch - start_epoch
  return int(min(start_beam_width + num_epochs / growth, end_beam_width))

per_epoch_metrics = []

for epoch in range(start_epoch, EPOCHS):
  start = time.time()
 
  sequence_loss_tot = 0
  token_loss_tot = 0
  num_steps = 0

  for (batch, (img_tensor, cap)) in enumerate(train_dataset):

    with writer.as_default():
      tf_summary('train_input_cap', cap, step=step_c, mode='')
   
      if epoch < XENT_EPOCHS or TRAIN_M == 0: 

        next_dec_fn = None
        if CONFIDENCE_SAMPLE_M == 0:
          next_dec_fn = sched_sampling
        elif CONFIDENCE_SAMPLE_M == 1:
          next_dec_fn = conf_sched_sampling 

        dec_input_fn = next_dec_fn if epoch > XENT_SCHED_EPOCHS else None
        sequence_loss, per_token_loss, train_vars, train_grads = train_step(img_tensor,
                                                                            cap,
                                                                            step_c,
                                                                            dec_input_fn=dec_input_fn)
      elif TRAIN_M == 1: 
        sequence_loss, per_token_loss, train_vars, train_grads = \
          train_step_bso(img_tensor,
                         cap,
                         encoder,
                         decoder,
                         step_c,
                         optimizer,
                         beam_width_fn(2, beam_width, 2, epoch, start_epoch),
                         vocab_size,
                         start_token_id=start_token_id,
                         eos_token_id=eos_token_id)
      elif TRAIN_M == 2: 
        sequence_loss, per_token_loss, train_vars, train_grads = \
          train_step_reinforce(img_tensor,
                               cap,
                               encoder,
                               decoder,
                               optimizer,
                               start_token = start_token_id,
                               eos_token = eos_token_id,
                               pad_token = 0)
      elif TRAIN_M == 3: 
        sequence_loss, per_token_loss, train_vars, train_grads = \
          train_step_self_critic(img_tensor,
                                 cap,
                                 encoder,
                                 decoder,
                                 optimizer,
                                 start_token_id,
                                 eos_token_id,
                                 vocab_size,
                                 pad_token = 0)
      elif TRAIN_M == 4: 
        sequence_reward_fn = SequencePerStepRewardFn(SequenceBleuRewardFn())
        # Wrap actor decoders and init delayed actor weights from actor weights
        actor_network = RnnDecoderNetwork(decoder) 

        if init_delayed_actor:
          print("decoder weights {} vs delayed weights {}".format(len(decoder.get_weights()), len(delayed_actor_decoder.get_weights())))
          delayed_actor_decoder.set_weights(decoder.get_weights())
          init_delayed_actor = False

        delayed_actor_network = RnnDecoderNetwork(delayed_actor_decoder) 
        actor_loss_weight = 1.0
        if epoch < CRITIC_FREEZE_EPOCHS:
          actor_loss_weight = 0.0
          print("Freeze actor at epoch {}".format(epoch))

        # Wrap critic decoders
        critic_network = RnnDecoderNetwork(critic_decoder) 
        if init_target_critic:
          target_critic_decoder.set_weights(critic_decoder.get_weights())
          init_target_critic = False

        target_critic_network = RnnDecoderNetwork(target_critic_decoder) 
 
        sequence_loss, per_token_loss, train_vars, train_grads = \
          train_step_actor_critic(img_tensor,
                                  cap,
                                  critic_network=critic_network,
                                  embedding_layer=embedding_layer,
                                  target_critic_network=target_critic_network,
                                  actor_network=actor_network,
                                  delayed_actor_network=delayed_actor_network,
                                  src_sequence_encoder=encoder,
                                  target_sequence_encoder=target_sequence_encoder,
                                  sequence_reward_fn=sequence_reward_fn,
                                  actor_optimizer=optimizer,
                                  critic_optimizer=critic_optimizer,
                                  start_token=start_token_id,
                                  eos_token=eos_token_id,
                                  vocab_size=vocab_size,
                                  actor_loss_weight=actor_loss_weight,
                                  pad_token = 0,
                                  step=step_c)
 
      tf_summary('train_sequence_loss', sequence_loss, step=step_c, mode='scalar')
      tf_summary('train_token_loss', per_token_loss, step=step_c, mode='scalar')
      for tidx, train_var in enumerate(train_vars):
        tf_summary('train_var_{}'.format(tidx), train_var, step=step_c, mode='histogram')
      for gidx, gr in enumerate(train_grads):
        tf_summary('train_grad_{}'.format(gidx), gr, step=step_c, mode='histogram')
      sequence_loss_tot += sequence_loss 
      token_loss_tot += per_token_loss 
      writer.flush()
      num_steps+=1
      logging.info("epoch {} batch {} sequence loss {} token loss {}".format(epoch+1, batch, sequence_loss.numpy(), per_token_loss.numpy()))
      step_c+=1
    
    if batch % 100 == 0:
      average_batch_loss = sequence_loss_tot.numpy()/int(num_steps)
      average_token_loss = token_loss_tot.numpy()/int(num_steps)
      print('Epoch {} Batch {} Sequence Loss {} Token Loss {}'.format(epoch+1, batch, average_batch_loss, average_token_loss))
  
  loss_plot.append(sequence_loss_tot/ num_steps)
 
  if epoch % 5 == 0:
    ckpt_manager.save()

  eval_cond = (epoch > EVAL_EPOCHS and (epoch - EVAL_EPOCHS) % EVAL_EVERY_EPOCHS == 0)

  if not eval_cond:
    print("Skip evaluation at epoch {}".format(epoch+1))
    continue

  bleu_scores = []

  print("Start evaluation at epoch {}".format(epoch+1))

  for (batch, (img_tensor, cap)) in enumerate(dev_dataset):
    with writer.as_default():
      tf_summary('dev_input_cap', cap, step=step_e, mode='')

      predict_sequence =\
        predict_image_caption_sequence(img_tensor,
                                       beam_width,
                                       max_length,
                                       step_e,
                                       training=False)
      predict_sequence_vec = []
      for pid_vec in predict_sequence:
        pid_vec = [pid.numpy() for pid in pid_vec]
        predict_sequence_vec.append(pid_vec) 
      predict_sequence_vec_shape = np.shape(predict_sequence_vec)
      b_size = predict_sequence_vec_shape[1]
      s_len = predict_sequence_vec_shape[0]
      predict_sequence_vec = np.reshape(predict_sequence_vec, [b_size, s_len])
      logging.debug("predict_sequence {}".format(np.shape(predict_sequence_vec)))
      logging.debug("cap {}".format(cap.shape))
      for in_batch_idx, cap_vec in enumerate(cap):
        cap_vec = [i.numpy() for i in cap_vec]
        pid_vec = predict_sequence_vec[in_batch_idx]
        logging.debug("cap vec {}".format(cap_vec)) 
        logging.debug("pid vec {}".format(pid_vec))
        pid_vec = [int(pid) for pid in pid_vec]
        result = [tokenizer.index_word[pid] for pid in pid_vec if
                  tokenizer.index_word[pid] not in ['s>', '<s>', '<pad>']]
        real = [tokenizer.index_word[i] for i in cap_vec if 
                tokenizer.index_word[i] not in ['<pad>', '<s>', 's>']]
        bleu_score = nltk.translate.bleu_score.sentence_bleu([real], result)
        bleu_scores.append(bleu_score)
        print("dev_set predict sequence {} predict cap {} real cap {} bleu {}"
              .format(predict_sequence, result, real, bleu_score))
      writer.flush()
      step_e+=1

  print('Epoch {} Loss {} bleu_scores {}'.format(epoch+1,
                                                 sequence_loss_tot / num_steps,
                                                 stats.describe(bleu_scores)))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
  
  avg_bleu_score = np.mean(bleu_scores)
  per_epoch_metrics.append((epoch+1, avg_bleu_score))
  

plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.savefig("train_loss.png")
plt.close()

epochs = []
bleus = []
for (epoch, bleu) in per_epoch_metrics:
  epochs.append(epoch)
  bleus.append(bleu)
plt.plot(epochs, bleus)
plt.xlabel('Epochs')
plt.ylabel('Dev Bleu')
plt.title('Dev Bleu Plot')
plt.savefig("dev_bleu.png")
plt.close()

print("Per epoch metrics {}".format(per_epoch_metrics))

# Model
## Prediction & Evaluation

def plot_attention(image, result, attention_plot, out_file):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(np.ceil(len_result/2), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.savefig(out_file)

test_dataset, max_test_length = create_test_dataset(test_img_name_vec,
                                                    test_captions,
                                                    test_batch_size=TEST_BATCH_SIZE)

bleu_scores = []

print("Start testing")

step_t = 0
for (batch, (img_tensor, cap)) in enumerate(test_dataset):

  with writer.as_default():
    tf_summary('test_input_cap', cap, step=step_t, mode='')

    predict_sequence =\
      predict_image_caption_sequence(img_tensor,
                                     beam_width,
                                     max_test_length,
                                     step_t,
                                     training=False)
    predict_sequence_vec = []
    for pid_vec in predict_sequence:
      pid_vec = [pid.numpy() for pid in pid_vec]
      predict_sequence_vec.append(pid_vec) 
    predict_sequence_vec_shape = np.shape(predict_sequence_vec)
    b_size = predict_sequence_vec_shape[1]
    s_len = predict_sequence_vec_shape[0]
    predict_sequence_vec = np.reshape(predict_sequence_vec, [b_size, s_len])
    logging.debug("predict_sequence {}".format(np.shape(predict_sequence_vec)))
    logging.debug("cap {}".format(cap.shape))
 
    for in_batch_idx, cap_vec in enumerate(cap):
      cap_vec = [i.numpy() for i in cap_vec]
      pid_vec = predict_sequence_vec[in_batch_idx]
      pid_vec = [int(pid) for pid in pid_vec]
      result = [tokenizer.index_word[pid] for pid in pid_vec
                if tokenizer.index_word[pid] not in ['s>', '<s>', '<pad>']]
      real = [tokenizer.index_word[i] for i in cap_vec 
              if tokenizer.index_word[i] not in ['s>', '<s>', '<pad>']]
      logging.debug("test_set predict sequence {} predict cap {} real cap {}"
            .format(predict_sequence, result, real))
      bleu_score = nltk.translate.bleu_score.sentence_bleu([real], result)
      bleu_scores.append(bleu_score)
      logging.debug('Bleu Score:', str(bleu_score))
    writer.flush()
    step_t+=1

print("TEST bleu {}".format(stats.describe(bleu_scores)))
