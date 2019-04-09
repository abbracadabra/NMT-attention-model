import tensorflow as tf
import numpy as np
from config import *

with tf.variable_scope('net_encode'):
    ph_src_embedding = tf.placeholder(dtype=tf.float32,shape=[src_vocab_size,src_w2v_dim],name='src_vocab_embedding_placeholder')
    #src_word_emb = tf.Variable(initial_value=ph_src_embedding,dtype=tf.float32,trainable=False, name='src_vocab_embedding_variable')

    encoder_X_ix = tf.placeholder(shape=(None, None), dtype=tf.int32)
    encoder_X_len = tf.placeholder(shape=(None), dtype=tf.int32)
    encoder_timestep = tf.shape(encoder_X_ix)[1]
    encoder_X = tf.nn.embedding_lookup(ph_src_embedding, encoder_X_ix)
    batchsize = tf.shape(encoder_X_ix)[0]

    encoder_Y_ix = tf.placeholder(shape=[None, None],dtype=tf.int32)
    encoder_Y_onehot = tf.one_hot(encoder_Y_ix, src_vocab_size)
    # encoder_Y_ix_reverse = tf.placeholder(shape=[None, None], dtype=tf.int32)
    # encoder_Y_onehot_reverse = tf.one_hot(encoder_Y_ix_reverse, src_vocab_size)

    enc_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(state_size, name='layer' + str(i)) for i in range(num_layers)])
    enc_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(state_size, name='layer' + str(i)) for i in range(num_layers)])
    enc_fw_initstate = enc_cell_fw.zero_state(batchsize,dtype=tf.float32)
    enc_bw_initstate = enc_cell_bw.zero_state(batchsize, dtype=tf.float32)
    enc_outputs, enc_final_states = tf.nn.bidirectional_dynamic_rnn(enc_cell_fw,enc_cell_bw,encoder_X,encoder_X_len,enc_fw_initstate,enc_bw_initstate)
    enc_fw_bw_outputs = tf.concat(enc_outputs, 2)
    enc_pred = tf.layers.dense(enc_fw_bw_outputs, units=src_vocab_size)
    # enc_fw_pred = tf.layers.dense(enc_outputs[0],units=src_vocab_size)
    # enc_bw_pred = tf.layers.dense(enc_outputs[1], units=src_vocab_size)
    encoder_loss = tf.losses.softmax_cross_entropy(encoder_Y_onehot,enc_pred)
    #encoder_bw_loss = tf.losses.softmax_cross_entropy(encoder_Y_onehot_reverse,enc_bw_pred)
    #encoder_loss = encoder_fw_loss+encoder_bw_loss
    fw_bw_final_cellstate = tf.concat([enc_final_states[0][-1].c,enc_final_states[1][-1].c],axis=-1)
    fw_bw_final_hidddenstate = tf.concat([enc_final_states[0][-1].h, enc_final_states[1][-1].h], axis=-1)
    encoder_trainop = tf.train.AdamOptimizer(0.001).minimize(encoder_loss)

with tf.variable_scope('net_decode'):
    ph_tgt_embedding = tf.placeholder(dtype=tf.float32, shape=[tgt_vocab_size, tgt_w2v_dim],
                                      name='tgt_vocab_embedding_placeholder')
    #tgt_word_emb = tf.Variable(initial_value=ph_tgt_embedding, dtype=tf.float32, trainable=False, name='tgt_vocab_embedding_variable')
    decoder_X_ix = tf.placeholder(shape=(None, None), dtype=tf.int32)
    decoder_timestep = tf.shape(decoder_X_ix)[1]
    decoder_X_len = tf.placeholder(shape=(None), dtype=tf.int32)
    decoder_X = tf.nn.embedding_lookup(ph_tgt_embedding, decoder_X_ix)

    decoder_Y_ix = tf.placeholder(shape=[None, None],dtype=tf.int32)
    decoder_Y_onehot = tf.one_hot(decoder_Y_ix, tgt_vocab_size)

    dec_cell = tf.nn.rnn_cell.LSTMCell(state_size*2)
    dec_initstate = tf.nn.rnn_cell.LSTMStateTuple(fw_bw_final_cellstate,fw_bw_final_hidddenstate)
    dec_outputs, dec_final_state = tf.nn.dynamic_rnn(dec_cell,decoder_X,decoder_X_len,dec_initstate)

    tile_enc = tf.tile(tf.expand_dims(enc_fw_bw_outputs,1),[1,decoder_timestep,1,1]) # [batchsize,decoder_len,encoder_len,state_size*2]
    tile_dec = tf.tile(tf.expand_dims(dec_outputs, 2), [1, 1, encoder_timestep, 1]) # [batchsize,decoder_len,encoder_len,state_size*2]
    enc_dec_cat = tf.concat([tile_enc,tile_dec],-1) # [batchsize,decoder_len,encoder_len,state_size*2*2]
    weights = tf.nn.softmax(tf.layers.dense(enc_dec_cat,units=1),axis=-2) # [batchsize,decoder_len,encoder_len,1]
    weighted_enc = tf.tile(weights, [1, 1, 1, state_size * 2])*tf.tile(tf.expand_dims(enc_fw_bw_outputs,1),[1,decoder_timestep,1,1]) # [batchsize,decoder_len,encoder_len,state_size*2]
    attentions = tf.reduce_sum(weighted_enc,axis=2,keepdims=False) # [batchsize,decoder_len,state_size*2]
    dec_attention_cat = tf.concat([dec_outputs,attentions],axis=-1) # [batchsize,decoder_len,state_size*2*2]
    dec_pred = tf.layers.dense(dec_attention_cat,units=tgt_vocab_size) # [batchsize,decoder_len,tgt_vocab_size]
    pred_ix = tf.argmax(dec_pred,axis=-1) # [batchsize,decoder_len]
    decoder_loss = tf.losses.softmax_cross_entropy(decoder_Y_onehot,dec_pred)
    total_loss = encoder_loss + decoder_loss
    decoder_trainop = tf.train.AdamOptimizer(0.001).minimize(total_loss)

_l0 = tf.summary.scalar('decoder_loss',decoder_loss)
_l1 = tf.summary.scalar('encoder_loss',encoder_loss)
log_all = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())
