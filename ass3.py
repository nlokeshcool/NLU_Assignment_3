from itertools import chain
import nltk
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import re
import string

sentences = []
single_sent = []
with open("ner.txt","r", encoding = "ISO-8859-1") as file:
    for line in file:
        if not (line and line.strip()):
            sentences.append(single_sent)
            single_sent = []
            continue
        tokens = nltk.word_tokenize(line)
        tuple_temp = ()
        for token in tokens:
            tuple_temp += (token,)
        single_sent.append(tuple_temp)


def word2features(sent, i):
    word = sent[i][0]
    special = 'False'
    if re.match(r'^\w+$', word):
        special = 'True'
    first_capital = word[0].isupper()
    prefix = word[0]
    if len(word) > 1:
        prefix += word[1]
    suffix = word[-1]
    if len(word) > 1:
        suffix += word[-1]
        
    ortho = ""
    if any(letter.isupper() for letter in word):
        ortho += "X"
    if any(letter.islower() for letter in word):
        ortho += "x"
    if any(not letter.isalpha() for letter in word):
        ortho += "$"
    
    features = [
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.pos=%s' % nltk.pos_tag(nltk.word_tokenize(word)),
        'word.special=%s' % special,
        'word.firstcap=%s' % first_capital,
        'word.prefix=%s' % prefix,
        'word.suffix=%s' % suffix,
        'word.ortho=%s' % ortho, 
        
    ]
    if i==0:
        features.extend(['word.begin=%s' % "Begin"])
    elif i == len(sent) - 1:
        features.extend(['word.begin=%s' % "End"])
    else:
        features.extend(['word.begin=%s' % "Middle"])
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

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



kf = KFold(n_splits = 5)
for train, test in kf.split(sentences):
    X_train = [sent2features(sentences[s]) for s in train]
    y_train = [sent2labels(sentences[s]) for s in train]
    
    X_test = [sent2features(sentences[s]) for s in test]
    y_test = [sent2labels(sentences[s]) for s in test]
    
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
    trainer.train('ner.crfsuite')
    
    tagger = pycrfsuite.Tagger()
    tagger.open('ner.crfsuite')

    
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    print(bio_classification_report(y_test, y_pred))
    
    """
    example_sent = test_sents[0]
    print(' '.join(sent2tokens(example_sent)), end='\n\n')
    
    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))
    
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    print(bio_classification_report(y_test, y_pred))
    """
    