import os
import re
import pickle
import nltk
import numpy as np
import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stop = stopwords.words('english')

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def clean_document(document):
    # Remove all characters outside of Alpha Numeric
    # and some punctuation
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = document.replace('-', '')
    document = document.replace('...', '')
    document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

    # Remove Ancronymns M.I.T. -> MIT
    # to help with sentence tokenizing
    document = merge_acronyms(document)

    # Remove extra whitespace
    document = ' '.join(document.split())
    return document

def remove_stop_words(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def similarity_score(t, s):
    t = remove_stop_words(t.lower())
    s = remove_stop_words(s.lower())
    t_tokens, s_tokens = t.split(), s.split()
    similar = [w for w in s_tokens if w in t_tokens]
    score = (len(similar) * 0.1 ) / len(t_tokens)
    return score

def merge_acronyms(s):
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.',''))
    return s

def rank_sentences(doc, doc_matrix, feature_names, top_n=3):
    sents = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS] for sent in sentences]
    tfidf_sent = [[doc_matrix[feature_names.index(w.lower())] for w in sent if w.lower() in feature_names]
                 for sent in sentences]

    # Calculate Sentence Values
    doc_val = sum(doc_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

    # Apply Similariy Score Weightings
    similarity_scores = [similarity_score(title, sent) for sent in sents]
    scored_sents = np.array(sent_values) + np.array(similarity_scores)

    # Apply Position Weights
    ranked_sents = [sent*(i/len(sent_values)) for i, sent in enumerate(sent_values)]

    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)

    return ranked_sents[:top_n]

if __name__ == '__main__':
    pass
