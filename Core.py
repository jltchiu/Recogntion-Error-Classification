import string, sys, os, math, array, re, operator
from collections import defaultdict as dd

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import jsonrpc
from simplejson import loads
def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def word2features(sent, i):
    word = sent[i][0]
    bursttag = sent[i][1]
    lastburst = sent[i][2]
    nextburst = sent[i][3]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'bursttag=' + bursttag,  #For baseline, please comment this line
    ]
    if i > 0:
        word1 = sent[i-1][0]
        bursttag1 = sent[i-1][1]
        lastburst1 = sent[i-1][2]
        nextburst1 = sent[i-1][3]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:bursttag=' + bursttag1,  #For baseline, please comment this line
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        bursttag1 = sent[i+1][1]
        lastburst1 = sent[i+1][2]
        nextburst1 = sent[i+1][3]        
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:bursttag=' + bursttag1, #For baseline, please comment this line
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, bursttag, lastburst, nextburst, label in sent]


action = sys.argv[1]


if action == "prep":
    ifile = open(sys.argv[2],"r")
    ofile = open(sys.argv[3],"w")
    lines = ifile.readlines()
    flag = 0
    text = []
    stoplist = []
    wordcount = dd(int)
    for l in lines:
        token = l.split()
        if len(token) < 2 or token[0] != "HYP:":
            continue
        for t in token:
            if "*" in t or "<" in t:
                continue
            wordcount[t.lower()] = wordcount[t.lower()] + 1     
        text.append(l)
    sorted_x = sorted(wordcount, key=wordcount.get, reverse=True)
    for i in range(len(sorted_x)/100):
        stoplist.append(sorted_x[i])
    for i in range(len(text)):
        token = text[i].split()[1:]
        token_last = []
        token_next = []
        if i > 0:
            token_last = text[i-1].split()[1:]
        if i < (len(text) - 1):
            token_next = text[i+1].split()[1:]
        for t in token:
            if "*" in t or "<" in t:
                continue
            if t.isupper():
                label = "ERROR"
            else:
                label = "CORRECT"
            if t.lower() not in stoplist:
                single_word = t + "," + str([x.lower() for x in token].count(t.lower()) > 1) + "," + str([y.lower() for y in token_last].count(t.lower()) > 0) + "," + str([z.lower() for z in token_next].count(t.lower()) > 0) + "," + label
            else:
                single_word = t + "," + "stop" + "," + "stop" + "," + "stop" + "," + label
            ofile.write(single_word+"\n")
        ofile.write("\n")
    ifile.close()
    ofile.close()



if action == "train":
    ifile = open(sys.argv[2],"r")
    ofilename = sys.argv[3]
    train_sents = []
    train_sent = []
    lines = ifile.readlines()
    for l in lines:
        if len(l.strip()) == 0:
            train_sents.append(tuple(train_sent))
            train_sent = []
            continue
        token = l.split(",")
        word = []
        for t in token:
            word.append(t)
        train_sent.append(word)

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.params()
    trainer.train(ofilename)

if action == "test":
    modelname = sys.argv[2]
    ifile = open(sys.argv[3],"r")
    test_sents = []
    test_sent = []
    lines = ifile.readlines()
    for l in lines:
        if len(l.strip()) == 0:
            test_sents.append(tuple(test_sent))
            test_sent = []
            continue
        token = l.split(",")
        word = []
        for t in token:
            word.append(t)
        test_sent.append(word)

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    tagger = pycrfsuite.Tagger()
    tagger.open(modelname)
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    print(bio_classification_report(y_test, y_pred))




