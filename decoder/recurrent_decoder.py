import tensorflow as tf

from utils.tf_utils import get_kwargs
from decoder.decoder import Decoder
from attention.attention import BahdanauAttention
from decoder.init_state import FeatureProjState 


class RecurrentDecoder(Decoder):

  def __init__(self,
               embedding_layer,
               embedding_dim,
               units,
               vocab_size,
               reset_state_from_feat = 0,
               data_type = tf.float32,
               attention_dropout = 0.0,
               recurrent_input_dropout = 0.0,
               recurrent_state_dropout = 0.0,
               recurrent_initializer='glorot_uniform',
               fc1_dropout=0.0,
               name='gru_decoder'):
     super(RecurrentDecoder, self).__init__(vocab_size, data_type, name=name)

     self.embedding_layer = embedding_layer
     self.embedding_dim = embedding_dim 
     self.units = units

     self.gru = tf.keras.layers.GRUCell(self.units,
                                        dropout=recurrent_input_dropout,
                                        recurrent_dropout=recurrent_state_dropout,
                                        recurrent_initializer=recurrent_initializer,
                                        name='{}/gru'.format(self.name))

     self.reset_state_from_feat = reset_state_from_feat 

     self.fc1 = tf.keras.layers.Dense(self.units, name='{}/fc1'.format(self.name))
     self.fc2 = tf.keras.layers.Dense(vocab_size, name='{}/fc2'.format(self.name))

     self.fc1_dropout = tf.keras.layers.Dropout(fc1_dropout)

     self.attention = BahdanauAttention(self.units,
                                        attention_dropout=attention_dropout,
                                        name='{}/attention'.format(self.name))

     self.fc3 = tf.keras.layers.Dense(1, name='{}/fc3'.format(self.name))

     self.num_cells = 1
     
     self.feat_proj_state_vec = [FeatureProjState(units,
       '{}/recurrent_feat_proj_{}'.format(self.name, hidx)) for hidx in range(self.num_cells)]

     self.args = get_kwargs()

  def call(self, inputs, **kwargs):
    dec_input, feat_d, hiddens = inputs

    features = feat_d['features']
    feature_bias = feat_d.get('bias')
    hidden = hiddens[0]

    training = kwargs['training'] 

    # dec_input: [batch_size, 1]
    # features: [batch_size, feature length, feature_dim]
    # hidden: [batch_size, hidden_size]

    # print("decode input {} features {} hidden {}".format(dec_input,
    # features, hidden))

    # dec_input: [batch_size]
    # dec_input_emb: [batch_size, embedding_dim]
    dec_input = tf.reshape(dec_input, [-1])
    dec_input_emb = self.embedding_layer(dec_input, **kwargs)

    attention_query = hidden
    
    #attention_query = tf.concat([attention_query, tf.reshape(dec_input_emb, [-1, dec_input_emb.shape[-1]])], -1)

    # context_vector: [batch_size, units]
    context_vector, attention_weights = self.attention(features,
                                                       attention_query,
                                                       training=training,
                                                       bias=feature_bias)


    # beta: [batch_size, 1]
    beta_coef = 1.0
    beta = tf.nn.sigmoid(self.fc3(hidden)) * beta_coef

    context_vector *= beta

    # print("decode attention vector {} weights {}".format(
    # context_vector, attention_weights))

    gru_input = tf.concat([context_vector, dec_input_emb], axis=-1)

    output, state = self.gru(gru_input, hidden, training=training)

    print("decode gru input {} output {} state {}".format(gru_input.shape, output.shape, state.shape))

    output = self.fc1(output)

    output = self.fc1_dropout(output, training=training)

    #output = tf.reshape(output, [-1, output.shape[-1]])

    print("gru output {} state {}".format(output.shape, state.shape))

    output_v = self.fc2(output)

    # print("decode output {}".format(output_v))

    return output_v, [state], attention_weights, context_vector

  def reset_state(self, target, feat_d, **kwargs):
    features = feat_d['features']
    feature_bias = feat_d.get('bias')
    batch_size = target.shape[0]
    initial_state = [tf.zeros((batch_size, self.units), dtype=self.data_type)] * self.num_cells
    out_initial_state = []
    for h_idx, h_state in enumerate(initial_state):
        feat_proj = self.feat_proj_state_vec[h_idx](features, mask=feature_bias)
        h_state += self.reset_state_from_feat * feat_proj
        out_initial_state.append(h_state)
    return out_initial_state
     
  def copy(self):
    return type(self)(**self.args)

  def create_variables(self):
    dec_input = tf.constant([0])
    hidden = self.reset_state(dec_input, {})
    return self.call((dec_input, hidden), training=False)
