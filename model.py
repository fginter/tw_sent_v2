import tensorflow as tf


import sys
import argparse
import data

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Activation, Conv1D, TimeDistributed, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Bidirectional, Concatenate,Flatten,Reshape,Dropout
from keras.optimizers import SGD, Adam
from keras.initializers import Constant

if tf.test.gpu_device_name():
    from keras.layers import CuDNNLSTM as LSTM
    print("Imported CuDNNLSTM",file=sys.stderr,flush=True)
else:
    from keras.layers import LSTM
    print("Imported standard LSTM",file=sys.stderr,flush=True)
    
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.regularizers import l2 as L2Reg


import numpy
import json
import re
import h5py

import random


class Classifier:

    def save_model(self,file_name):
        model_json = self.model.to_json()
        with open(file_name+".model.json","w") as f:
            print(model_json,file=f)

    
            
class DocuClassifier(Classifier):

    def __init__(self):
        pass

    def build_model(self, cls_count, word_seq_len, word_vec, lemma_vec, **kwargs):
        rnn_dim=200
        dense_dim=200
        kern_l2=L2Reg(kwargs.get("kern_l2",0.0))
        act_l2=L2Reg(kwargs.get("act_l2",0.0))
        dr=kwargs.get("dr",0.0)
        lr=kwargs.get("lr",0.001)
        
        inp_words=Input(name="inp_words",shape=(word_seq_len,))
        inp_lemmas=Input(name="inp_lemmas",shape=(word_seq_len,))

        word_emb=Embedding(word_vec.vectors.shape[0],word_vec.vectors.shape[1],name="emb_word",trainable=False,mask_zero=False,weights=[word_vec.vectors])(inp_words)
        lemma_emb=Embedding(lemma_vec.vectors.shape[0],lemma_vec.vectors.shape[1],name="emb_lemma",trainable=False,mask_zero=False,weights=[lemma_vec.vectors])(inp_lemmas)
        
        rnn_word_seq=Bidirectional(LSTM(rnn_dim,kernel_regularizer=kern_l2,activity_regularizer=act_l2,return_sequences=True))(Dropout(rate=dr)(word_emb))
        rnn_lemma_seq=Bidirectional(LSTM(rnn_dim,kernel_regularizer=kern_l2,activity_regularizer=act_l2,return_sequences=True))(Dropout(rate=dr)(lemma_emb))

        rnn_word_out=GlobalMaxPooling1D()(rnn_word_seq)
        rnn_lemma_out=GlobalMaxPooling1D()(rnn_lemma_seq)
        
        cc=Concatenate()([rnn_word_out,rnn_lemma_out])
        hidden=Dense(dense_dim, activation="tanh")(cc)

        outp=Dense(cls_count,name="joint",activation="softmax")(hidden)

        
        self.model=Model(inputs=[inp_words,inp_lemmas], outputs=[outp])
        self.optimizer=Adam(lr,amsgrad=True)
        self.model.compile(optimizer=self.optimizer,metrics=['accuracy'],loss="sparse_categorical_crossentropy")

class DocuClassifierMultiout(Classifier):

    def __init__(self):
        pass


    def build_model(self, label_encoders, word_seq_len, word_vec, lemma_vec, **kwargs):
        rnn_dim=200
        dense_dim=200
        kern_l2=L2Reg(kwargs.get("kern_l2",0.0))
        act_l2=L2Reg(kwargs.get("act_l2",0.0))
        dr=kwargs.get("dr",0.0)
        lr=kwargs.get("lr",0.001)
        
        inp_words=Input(name="inp_words",shape=(word_seq_len,))
        inp_lemmas=Input(name="inp_lemmas",shape=(word_seq_len,))

        word_emb=Embedding(word_vec.vectors.shape[0],word_vec.vectors.shape[1],name="emb_word",trainable=False,mask_zero=False,weights=[word_vec.vectors])(inp_words)
        lemma_emb=Embedding(lemma_vec.vectors.shape[0],lemma_vec.vectors.shape[1],name="emb_lemma",trainable=False,mask_zero=False,weights=[lemma_vec.vectors])(inp_lemmas)
        
        rnn_word_seq=Bidirectional(LSTM(rnn_dim,kernel_regularizer=kern_l2,activity_regularizer=act_l2,return_sequences=True))(Dropout(rate=dr)(word_emb))
        rnn_lemma_seq=Bidirectional(LSTM(rnn_dim,kernel_regularizer=kern_l2,activity_regularizer=act_l2,return_sequences=True))(Dropout(rate=dr)(lemma_emb))

        rnn_word_out=GlobalMaxPooling1D()(rnn_word_seq)
        rnn_lemma_out=GlobalMaxPooling1D()(rnn_lemma_seq)
        
        cc=Concatenate()([rnn_word_out,rnn_lemma_out])
        hidden=Dense(dense_dim, activation="tanh")(cc)

        outp_joint=Dense(len(label_encoders["joint"].classes_),name="joint",activation="softmax")(hidden)
        outp_aspect=Dense(len(label_encoders["aspect"].classes_),name="aspect",activation="softmax")(hidden)
        outp_sentiment=Dense(len(label_encoders["sentiment"].classes_),name="sentiment",activation="softmax")(hidden)

        self.model=Model(inputs=[inp_words,inp_lemmas], outputs=[outp_aspect])
        self.optimizer=Adam(lr,amsgrad=True)
        self.model.compile(optimizer=self.optimizer,metrics=['accuracy'],loss="sparse_categorical_crossentropy")
        
        
class DocuClassifierMultioutCNN(Classifier):

    def __init__(self):
        pass

    def build_model(self, label_encoders, word_seq_len, word_vec, lemma_vec, **kwargs):
        rnn_dim=600
        dense_dim=200
        kern_l2_val=kwargs.get("kern_l2",0.0)
        kern_l2=L2Reg(kern_l2_val)
        act_l2=L2Reg(kwargs.get("act_l2",kern_l2_val))
        dr=kwargs.get("dr",0.0)
        lr=kwargs.get("lr",0.001)
        
        inp_words=Input(name="inp_words",shape=(word_seq_len,))
        inp_lemmas=Input(name="inp_lemmas",shape=(word_seq_len,))

        word_emb=Embedding(word_vec.vectors.shape[0],word_vec.vectors.shape[1],name="emb_word",trainable=False,mask_zero=False,weights=[word_vec.vectors])(inp_words)
        lemma_emb=Embedding(lemma_vec.vectors.shape[0],lemma_vec.vectors.shape[1],name="emb_lemma",trainable=False,mask_zero=False,weights=[lemma_vec.vectors])(inp_lemmas)

        rnn_word_seq_1=GlobalMaxPooling1D()(Conv1D(rnn_dim,1,kernel_regularizer=kern_l2,activity_regularizer=act_l2,padding="same")(word_emb))
        rnn_lemma_seq_1=GlobalMaxPooling1D()(Conv1D(rnn_dim,1,kernel_regularizer=kern_l2,activity_regularizer=act_l2,padding="same")(lemma_emb))
        rnn_word_seq_2=GlobalMaxPooling1D()(Conv1D(rnn_dim,2,kernel_regularizer=kern_l2,activity_regularizer=act_l2,padding="same")(word_emb))
        rnn_lemma_seq_2=GlobalMaxPooling1D()(Conv1D(rnn_dim,2,kernel_regularizer=kern_l2,activity_regularizer=act_l2,padding="same")(lemma_emb))
        
        cc=Concatenate()([rnn_word_seq_1,rnn_lemma_seq_1,rnn_word_seq_2,rnn_lemma_seq_2])
        hidden=Dense(dense_dim, activation="tanh")(cc)

        outp_joint=Dense(len(label_encoders["joint"].classes_),name="joint",activation="softmax")(hidden)
        outp_aspect=Dense(len(label_encoders["aspect"].classes_),name="aspect",activation="softmax")(hidden)
        outp_sentiment=Dense(len(label_encoders["sentiment"].classes_),name="sentiment",activation="softmax")(hidden)
        
        self.model=Model(inputs=[inp_words,inp_lemmas], outputs=[outp_aspect,outp_sentiment,outp_joint])
        self.optimizer=Adam(lr,amsgrad=True)
        self.model.compile(optimizer=self.optimizer,metrics=['accuracy'],loss="sparse_categorical_crossentropy")
        
