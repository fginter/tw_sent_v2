import sys
import json
import random
import numpy
import numpy.random
import keras.utils
from keras.preprocessing.sequence import pad_sequences
import model
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab

ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)

def read_conll(inp,max_sent=0,drop_tokens=True,drop_nulls=True):
    comments=[]
    sent=[]
    yielded=0
    for line in inp:
        line=line.strip()
        if line.startswith("#"):
            comments.append(line)
        elif not line:
            if sent:
                yield sent,comments
                yielded+=1
                if max_sent>0 and yielded==max_sent:
                    break
                sent,comments=[],[]
        else:
            cols=line.split("\t")
            if drop_tokens and "-" in cols[ID]:
                continue
            if drop_nulls and "." in cols[ID]:
                continue
            sent.append(cols)
    else:
        if sent:
            yield sent,comments

            
def read_embeddings(embeddings_filename,max_rank_emb):
    """Reads .vector or .bin file, modifies it to include <OOV> and <PADDING> and <SENTROOT>"""
    if embeddings_filename.endswith(".bin"):
        binary=True
    else:
        binary=False
    gensim_vectors=KeyedVectors.load_word2vec_format(embeddings_filename, binary=binary, limit=max_rank_emb)
    gensim_vectors.vocab["<OOV>"]=Vocab(index=1)
    gensim_vectors.vocab["<PADDING>"]=Vocab(index=0)
    gensim_vectors.vocab["<SENTROOT>"]=Vocab(index=2)
    for word_record in gensim_vectors.vocab.values():
        word_record.index+=3
    two_random_rows=numpy.random.uniform(low=-0.01, high=0.01, size=(3,gensim_vectors.vectors.shape[1]))
    # stack the two rows, and the embedding matrix on top of each other
    gensim_vectors.vectors=numpy.vstack([two_random_rows,gensim_vectors.vectors])
    gensim_vectors.vectors=keras.utils.normalize(gensim_vectors.vectors,axis=0)
    gensim_vectors.vectors=keras.utils.normalize(gensim_vectors.vectors)
    return gensim_vectors

def build_dicts(inp):
    char_dict={"<PAD>":0,"<OOV>":1}
    pos_dict={"<OOV>":0}
    deprel_dict={"<OOV>":0}
    feat_val_dict={} #"number" ->  {"<UNSET>":0,"sg":1}
    for tree,comments in read_conll(inp):
        for cols in tree:
            for char in cols[FORM]:
                char_dict.setdefault(char,len(char_dict))
            pos_dict.setdefault(cols[UPOS],len(pos_dict))
            deprel_dict.setdefault(cols[DEPREL],len(deprel_dict))
            deprel_dict.setdefault(cols[DEPREL]+"-left",len(deprel_dict))
            deprel_dict.setdefault(cols[DEPREL]+"-right",len(deprel_dict))
            if cols[FEATS]!="_":
                for feat_val in cols[FEATS].split("|"):
                    feat,val=feat_val.split("=",1)
                    feat_dict=feat_val_dict.setdefault(feat,{"<UNSET>":0})
                    feat_dict.setdefault(val,len(feat_dict))
    return char_dict,pos_dict,deprel_dict,feat_val_dict


ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS=range(10)
def vectorize_document(doc,word_vocab,lemma_vocab):
    words=[]
    lemmas=[]
    #doc is a list of conllu sentences
    for sent,comment in doc:
        for cols in sent:
            if cols[FORM] in word_vocab:
                words.append(word_vocab[cols[FORM]].index)
            else:
                words.append(word_vocab["<OOV>"].index)
            if cols[LEMMA] in lemma_vocab:
                lemmas.append(lemma_vocab[cols[LEMMA]].index)
            else:
                lemmas.append(lemma_vocab["<OOV>"].index)
    return words, lemmas

def vectorize_data(docs, word_vocab, lemma_vocab):
    docs_words,docs_lemmas=[],[]
    for doc in docs:
         words, lemmas = vectorize_document(doc,word_vocab,lemma_vocab)
         docs_words.append(words)
         docs_lemmas.append(lemmas)
    docs_words=pad_sequences(np.array(docs_words),padding="post",maxlen=word_seq_len)
    docs_lemmas=pad_sequences(np.array(docs_lemmas),padding="post",maxlen=word_seq_len)
    assert docs_words.shape==docs_lemmas.shape
    return {"inp_words":docs_words,"inp_lemmas":docs_lemmas}

         
    
