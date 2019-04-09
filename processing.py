#!/usr/bin/env python
from config import *
from subprocess import run, check_output, DEVNULL

MOSES_BDIR = os.path.join(basedir,"moses",'tokenizer')
REM_NON_PRINT_CHAR = os.path.join(MOSES_BDIR ,'remove-non-printing-char.perl')
DESCAPE = os.path.join(MOSES_BDIR ,'deescape-special-chars.perl')
NORM_PUNC = os.path.join(MOSES_BDIR ,'normalize-punctuation.perl')
MOSES_LC = os.path.join(MOSES_BDIR ,'lowercase.perl')
MOSES_TOKENIZER = os.path.join(MOSES_BDIR ,'tokenizer.perl -q -no-escape -threads 20')
CJK_CHAR_SPLIT = os.path.join(MOSES_BDIR ,'cjk-char-split.py')

src_traindata_raw='~/wind/Users/yl_gong/PycharmProjects/rnnnnnn/data/news-commentary-v13.zh-en.en'
tgt_traindata_raw='~/wind/Users/yl_gong/PycharmProjects/rnnnnnn/data/news-commentary-v13.zh-en.zh'

def segmentation(txt_fp):
    run("cat "+txt_fp+' | '+REM_NON_PRINT_CHAR+'|'+NORM_PUNC+'|'+DESCAPE+'|'+MOSES_TOKENIZER+'|'+CJK_CHAR_SPLIT+'>'+txt_fp+'.tok',shell=True)

def subwordsegmentation(txt_fp):
    # apply bpe
    pass

if __name__ == '__main__':
    segmentation(src_traindata_raw)
    segmentation(tgt_traindata_raw)
