import numpy as np

def countline(fp):
    with open(fp,encoding='utf-8') as f:
        lc = 0
        for _ in f:
            lc += 1
    return lc


def loadw2v(w2v_fp):
    f = open(w2v_fp,'r',encoding='utf-8')
    w2i = {}
    i2w = {}
    w2v = []
    for i,line in enumerate(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        w2v.append(embedding)
        w2i[word] = i
        i2w[i] = word
    return np.array(w2v),w2i,i2w


def getbatchlen(batch):
    return [len(line_splits) for line_splits in batch]

def padline(line_splits, maxlen, tok):

    pass