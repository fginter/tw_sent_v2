from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np

import sys
import pickle
import os.path
import argparse
from data import get_class_from_comments, read_conll


def split(proportion,x,y,stratified=True):
    if stratified:
        splitter=StratifiedShuffleSplit(n_splits=1,test_size=1.0-proportion)
    else:
        splitter=ShuffleSplit(n_splits=1,test_size=1.0-proportion)
    indices0,indices1=list(splitter.split(x,y))[0]
    return x[indices0],y[indices0],x[indices1],y[indices1]


def print_data(x,y,data,f_name):
    with open(f_name,"w") as f:
        for data_idx,lab in zip(x,y):
            doclab,sentences=data[data_idx]
            for sent,comments in sentences:
                for c in comments:
                    print(c,file=f)
                for cols in sent:
                    print(*cols,sep="\t",file=f)
                print(file=f)


def read_docs(inp):
    documents=[]  #(class,[sentence,sentence,...])
    for tree, comments in read_conll(inp):
        cls=get_class_from_comments(comments) #cls is actually a list of classes
        if cls:
            documents.append((cls[0],[])) #just take the first class, there's like six multilabel examples
        documents[-1][1].append((tree,comments))
    print("len(documents)",len(documents),file=sys.stderr,flush=True)
    return documents


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('outname', nargs=1, help='Output file name, will create .train .dev .test')
    args = parser.parse_args()

    documents=read_docs(sys.stdin)

    labels=[item[0] for item in documents]
    label_encoder=LabelEncoder()
    label_encoder.fit_transform(labels)
    
    class_numbers=label_encoder.transform(labels)
    indices=np.arange(len(labels))

    traindev_x,traindev_y,test_x,test_y=split(0.9,indices,class_numbers,stratified=True) #0.1 test set
    train_x,train_y,dev_x,dev_y=split(0.9,traindev_x,traindev_y,stratified=False)

    assert set(train_x) & set(dev_x) & set(test_x)==set()
    assert len(train_x)+len(dev_x)+len(test_x)==len(labels)
    assert set(train_x)|set(dev_x)|set(test_x)==set(np.arange(len(labels)))

    print_data(train_x,train_y,documents,args.outname[0]+".train")
    print_data(dev_x,dev_y,documents,args.outname[0]+".dev")
    print_data(test_x,test_y,documents,args.outname[0]+".test")
    
