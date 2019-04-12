from model import *
from processing import *
from util import *
import copy
import time

def processbatch(strings,w2i,prepend):
    batchtokens = [s.split() for s in strings]
    if prepend==True:
        for toks in batchtokens:
            toks.insert(0,'<sos>')
    lens = getbatchlen(batchtokens)
    maxlen = np.max(lens)
    for line_toks in batchtokens:
        line_toks+=['<eos>']*(maxlen-len(line_toks))
    X_toks = copy.deepcopy(batchtokens)
    Y_toks = copy.deepcopy(batchtokens)
    Y_toks = [ts[1:] for ts in Y_toks]
    for line_toks in Y_toks:
        line_toks += ['<eos>']
    src_ix = []
    for line_toks in X_toks:
        src_ix.append([w2i[tk] if tk in w2i else w2i['<unk>'] for tk in line_toks])
    tgt_ix = []
    for line_toks in Y_toks:
        tgt_ix.append([w2i[tk] if tk in w2i else w2i['<unk>'] for tk in line_toks])
    return lens,src_ix,tgt_ix

def getbatch():
    fs = open(src_traindata_tokenized, encoding='utf-8')
    ft = open(tgt_traindata_tokenized, encoding='utf-8')
    encoder_batch = []
    encoder_batch_reversed = []
    decoder_batch = []
    for s, t in zip(fs, ft):
        encoder_batch.append(s)
        encoder_batch_reversed.append(s[::-1])
        decoder_batch.append(t)
        if len(encoder_batch)==batch_size:
            ec_X_lens, ec_X_ix, ec_Y_ix = processbatch(encoder_batch,src_w2i,prepend=False)
            #ec_X_lens, ec_X_ix_reverse, ec_Y_ix_reverse = processbatch(encoder_batch_reversed, src_w2i, prepend=False)
            dc_X_lens, dc_X_ix, dc_Y_ix = processbatch(decoder_batch, tgt_w2i,prepend=True)
            encoder_batch = []
            encoder_batch_reversed = []
            decoder_batch = []
            yield (ec_X_lens, ec_X_ix, ec_Y_ix),(dc_X_lens, dc_X_ix, dc_Y_ix)

if __name__ == '__main__':
    assert countline(src_traindata_tokenized) == countline(tgt_traindata_tokenized)
    src_w2v, src_w2i, src_i2w = loadw2v(src_lang_embedding)
    tgt_w2v, tgt_w2i, tgt_i2w = loadw2v(tgt_lang_embedding)

    saver = tf.train.Saver()
    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    starttime = time.time()
    for i in range(epochs):
        for j, ((ec_X_lens, ec_X_ix, ec_Y_ix), (dc_X_lens, dc_X_ix, dc_Y_ix)) in enumerate(getbatch()):
            _, _, encode_err, _log, pred = sess.run([encoder_trainop,decoder_trainop, total_loss, log_all, pred_ix],
                                                 feed_dict={encoder_X_len: ec_X_lens, encoder_X_ix: ec_X_ix,
                                                            encoder_Y_ix: ec_Y_ix,
                                                            decoder_X_len: dc_X_lens, decoder_X_ix: dc_X_ix,
                                                            decoder_Y_ix: dc_Y_ix,
                                                            # encoder_Y_ix_reverse:ec_Y_ix_reverse,
                                                            ph_src_embedding: src_w2v, ph_tgt_embedding: tgt_w2v})
            writer.add_summary(_log)
            print(encode_err)
            if j % 10 == 0:
                saver.save(sess, model_path)
                for l in pred:
                    ss = []
                    for ll in l:
                        ss.append(tgt_i2w[ll])
                    print(ss)

            if (time.time() - starttime) / 60 > 240:
                raise Exception('120min')



