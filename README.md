### an en-zh nmt model implemented with <a href='https://nlp.stanford.edu/pubs/emnlp15_attn.pdf'>Effective Approaches to Attention-based Neural Machine Translation</a>
#### model configuration
- <a href='https://nlp.stanford.edu/projects/glove/'>glove(50d)</a> word embedding for english language(vocab size 400k)
- <a href='https://github.com/Embedding/Chinese-Word-Vectors'>something i found</a> for chinese language word embedding(vocab size 20k)
- state size of 120
#### somethoughts
- In decoder phase,maybe we dont have to plug previous step prediction into next step.Namely,we can plug zeros as input to decoder.This way,in inference time,we can predict all words at once instead of step by step prediction.
