from trigram_tagger import SubjectTrigramTagger
from bs4 import BeautifulSoup
import requests
import re
import pickle
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Noun Part of Speech Tags used by NLTK
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def download_document(url):
    """Downloads document using BeautifulSoup, extracts the subject and all
    text stored in paragraph tags
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    title = soup.find('title').get_text()
    document = ' '.join([p.get_text() for p in soup.find_all('p')])
    return document

def clean_document(document):
    """Remove enronious characters. Extra whitespace and stop words"""
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def tokenize_sentences(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences

def get_entities(document):
    """Returns Named Entities using NLTK Chunking"""
    entities = []
    sentences = tokenize_sentences(document)

    # Part of Speech Tagging
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                entities.append(' '.join([c[0] for c in chunk]).lower())
    return entities

def word_freq_dist(document):
    words = nltk.tokenize.word_tokenize(document)
    words = [word.lower() for word in words if word not in stop]
    fdist = nltk.FreqDist(words)
    return fdist

def extract_subject(document):
    # Get most frequent Nouns
    fdist = word_freq_dist(document)
    most_freq_nouns = [w for w, c in fdist.most_common(10)
                       if nltk.pos_tag([w])[0][1] in NOUNS]

    # Get Top 10 entities
    entities = get_entities(document)
    top_10_entities = [w for w, c in nltk.FreqDist(entities).most_common(10)]

    # Get the subject noun by looking at the intersection of top 10 entities
    # and most frequent nouns. It takes the first element in the list
    subject_nouns = [entity for entity in top_10_entities
                    if entity.split()[0] in most_freq_nouns]
    return subject_nouns[0]

def trained_tagger(existing=False):
    if existing:
        trigram_tagger = picke.load(open('trained_tagger.pkl', 'rb'))
        return trigram_tagger

    # Aggregate trained sentences for N-Gram Taggers
    train_sents = nltk.corpus.brown.tagged_sents()
    train_sents += nltk.corpus.conll2000.tagged_sents()
    train_sents += nltk.corpus.treebank.tagged_sents()

    # Create instance of SubjectTrigramTagger and persist instance of it
    trigram_tagger = SubjectTrigramTagger(train_sents)
    pickle.dump(trigram_tagger, open('trained_tagger.pkl', 'wb'))

    return trigram_tagger

def tag_sentences(document):
    trigram_tagger = trained_tagger()

    # Tokenize Sentences and words
    sentences = tokenize_sentences(document)

    # Tag each sentence
    tagged_sents = [trigram_tagger.tag(sent) for sent in sentences]
    return tagged_sents

if __name__ == '__main__':
    # url = 'http://techcrunch.com/2016/05/26/snapchat-series-f/?ncid=tcdaily'
    # document = download_document(url)
    document = pickle.load(open('document.pkl', 'rb'))
    document = clean_document(document)
    subject = extract_subject(document)
    print subject

    tagged_sents = tag_sentences(document)
    print tagged_sents[:10]

    # extract_subject()
