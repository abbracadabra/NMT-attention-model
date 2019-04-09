from model import *
import numpy as np
from processing import *
from util import *
import copy

def evaluate(ix):
    initstate,encout = sess.run([dec_initstate,enc_fw_bw_outputs],
                                feed_dict={encoder_X_ix:[ix],encoder_X_len:[len(ix)],ph_src_embedding:src_w2v})
    pw = ''
    step = 0
    initdecix = tgt_w2i['<sos>']
    predsentence = ''
    while pw!='<eos>':
        pix,state = sess.run([pred_ix,dec_final_state],
                       feed_dict={dec_initstate:initstate,
                                  enc_fw_bw_outputs: encout,
                                  decoder_X_ix: [[initdecix]], decoder_X_len: [1],
                                  ph_tgt_embedding: tgt_w2v,
                                  encoder_timestep:len(encout[0])})
        pw = tgt_i2w[pix[0][0]]
        initstate = state
        initdecix = pix[0][0]
        predsentence += pw
        step+=1
    return predsentence

def gettest():
    f = open(test_tokenized, encoding='utf-8')
    for s in f:
        toks = s.split()
        ix = [src_w2i[tk] if tk in src_w2i else src_w2i['<unk>'] for tk in toks]
        yield ix

if __name__ == '__main__':
    src_w2v, src_w2i, src_i2w = loadw2v(src_lang_embedding)
    tgt_w2v, tgt_w2i, tgt_i2w = loadw2v(tgt_lang_embedding)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_path)

    for ix in gettest():
        print(evaluate(ix))








