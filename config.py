import os

basedir = os.getcwd()
src_traindata_tokenized = os.path.join(basedir,'data','news-commentary-v13.zh-en.en.tok')
tgt_traindata_tokenized = os.path.join(basedir,'data','news-commentary-v13.zh-en.zh.tok')
test_tokenized = os.path.join(basedir,'data','test.tok')
src_lang_embedding = r'D:\Users\yl_gong\Desktop\dl\glove.6B\glove.6B.50d.txt'
tgt_lang_embedding = r'D:\Users\yl_gong\Desktop\dl\sgns.sikuquanshu.word\sgns.sikuquanshu.word'
model_path = os.path.join(basedir,'mdl','seq2seq_attention')
log_path = os.path.join(basedir,'log')
src_w2v_dim=50
tgt_w2v_dim = 300
state_size=120
num_layers=2
src_vocab_size = 400003
tgt_vocab_size = 19530
epochs = 3
batch_size=2

