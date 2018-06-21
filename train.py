import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = -1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import sys
import argparse
import data
import split_data

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Activation, Conv1D, TimeDistributed, Flatten, GlobalMaxPooling1D
from keras.layers import Bidirectional, Concatenate,Flatten,Reshape,Dropout
from keras.optimizers import SGD, Adam
from keras.initializers import Constant
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.regularizers import l2 as L2Reg

from sklearn.preprocessing import LabelEncoder

import numpy
import json
import re
import h5py

import random
import model
import time
import gc

import rbfopt
import functools

def train(datasets,args,label_encoders,batch_size=100,epochs=100,patience=15,params_in_name=True,wipe_mem=True,**kwargs): #kwargs is parameters
    (train_X,dev_X,train_labels_numeric,dev_labels_numeric)=datasets
    param_string="__".join("{}_{}".format(k,v) for k,v in sorted(kwargs.items()))
    model_class=getattr(model,args.classname) #Pick the right class
    model_name=args.model_file
    if params_in_name:
        model_name+="__"+param_string
    m=model_class() #instantiate the model
    m.build_model(label_encoders, args.word_seq_len, word_vec, lemma_vec, **kwargs)
    m.save_model(model_name)
    save_cb=ModelCheckpoint(filepath=model_name+".weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es_cb=EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    tensorboard_log_dir="{}.tensorboard.log/{}".format(args.model_file,param_string)
    tb_cb=TensorBoard(tensorboard_log_dir)
    print("Tensorboard logs in", tensorboard_log_dir, file=sys.stderr)
    class_weights={}
    for k in train_labels_numeric:
        classes=numpy.unique(train_labels_numeric[k])
        class_weights[k]=dict((c,w) for c,w in zip(classes,class_weight.compute_class_weight('balanced', classes, train_labels_numeric[k])))
    hist=m.model.fit(x=train_X, y=train_labels_numeric, validation_data=(dev_X, dev_labels_numeric), verbose=1, batch_size=batch_size, epochs=epochs, class_weight=class_weights,callbacks=[save_cb,es_cb,tb_cb])
    with open(model_name+".history.json","w") as f:
        json.dump((hist.epoch,hist.history),f)
    retval=float(min(hist.history["val_loss"]))

    # m.model.load_weights(model_name+".weights.h5")
    
    # print("Network output=",m.model.predict(test_X))
    # predictions=dict((k,p) for k,p in zip(["aspect","sentiment","joint"],m.model.predict(test_X)))
    # for k in predictions:
    #     predictions_labels=label_encoders[k].inverse_transform(numpy.argmax(predictions[k],axis=1))
    #     gold_labels=label_encoders[k].inverse_transform(test_labels_numeric[k])
    #     unique_labels=list(set(gold_labels))
    #     print(unique_labels)
    #     CM=confusion_matrix(gold_labels, predictions_labels, unique_labels)
    #     maxlablen=max(len(l) for l in unique_labels)
    #     for lab,row in zip(unique_labels,CM):
    #         print(lab.ljust(maxlablen+5),*(str(v).ljust(5) for v in row))
    #     print(classification_report(gold_labels,predictions_labels))

    if wipe_mem:
        del m.model
        del m
        del hist
        clear_session()
        for _ in range(10):
            gc.collect()
            time.sleep(0.1)

    return retval

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--train-file', help='.conllu')
    parser.add_argument('--devel-file', help='.conllu')
    parser.add_argument('--test-file', help='.conllu')
    parser.add_argument('--word-embeddings', help='.vector or .bin')
    parser.add_argument('--lemma-embeddings', help='.vector or .bin')
    parser.add_argument('--maxrank-emb', type=int, default=100000, help='Max rank of the embedding')
    parser.add_argument('--classname', default="DocuClassifier", help='Name of class in model.py')
    parser.add_argument('--word-seq-len', type=int, default=400, help='Name of class in model.py')
    parser.add_argument('--model-file', help='file-name-prefix to save to')
    
    #parser.add_argument('--like', help='train a new model like this one, pick parameters from the name')
    
    args = parser.parse_args()

    word_seq_len=args.word_seq_len
    
    word_vec=data.read_embeddings(args.word_embeddings,args.maxrank_emb)
    lemma_vec=data.read_embeddings(args.lemma_embeddings,args.maxrank_emb)

    
    train_documents=split_data.read_docs(open(args.train_file))
    dev_documents=split_data.read_docs(open(args.devel_file))
    test_documents=split_data.read_docs(open(args.test_file))
    
    train_labels=[item[0] for item in train_documents]
    dev_labels=[item[0] for item in dev_documents]
    test_labels=[item[0] for item in test_documents]

    train_labels_numeric, label_encoders=data.vectorize_labels(train_labels,dict((k,None) for k in ("joint","aspect","sentiment")))
    #train_labels_numeric is a dictionary with joint,aspect,sentiment as keys and the encoded labels as values
    dev_labels_numeric,_=data.vectorize_labels(dev_labels,label_encoders)
    test_labels_numeric,_=data.vectorize_labels(test_labels,label_encoders)

    train_X=data.vectorize_data([item[1] for item in train_documents], word_seq_len, word_vec.vocab, lemma_vec.vocab)
    dev_X=data.vectorize_data([item[1] for item in dev_documents], word_seq_len, word_vec.vocab, lemma_vec.vocab)
    test_X=data.vectorize_data([item[1] for item in test_documents], word_seq_len, word_vec.vocab, lemma_vec.vocab)

    def train_func(params):
        lr,dr,reg=params
        return train((train_X,dev_X,train_labels_numeric, dev_labels_numeric), args, label_encoders,lr=lr,kern_l2=reg,dr=dr)

    #                                             lr       dout   reg
    bb=rbfopt.RbfoptUserBlackBox(3,numpy.array([0.000001,  0.0,   0.00000001]),\
                                 numpy.array([  0.001   ,  0.6,   0.0001]),numpy.array(['R','R','R']),train_func)
    settings = rbfopt.RbfoptSettings(max_clock_time=180*60,target_objval=0.0,num_cpus=1)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    val, x, itercount, evalcount, fast_evalcount = alg.optimize()


