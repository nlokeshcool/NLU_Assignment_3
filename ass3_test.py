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


with open("test.txt") as file:
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
    if len(single_sent) > 0:
        sentences.append(single_sent)
        
X_test = [sent2features(s) for s in sentences]


tagger = pycrfsuite.Tagger()
tagger.open('ner.crfsuite')


y_pred = [tagger.tag(xseq) for xseq in X_test]

index = 0
sent_index = 0
output = ""
with open("test.txt") as file:
    for line in file:
        if not (line and line.strip()):
            output += "\n"
            sent_index += 1
            index = 0
            continue
        output += str(line.strip()) + " " + str(y_pred[sent_index][index]) + "\n"
        index += 1
with open("output.txt", "w") as file:
    file.write(output)
        
            
        