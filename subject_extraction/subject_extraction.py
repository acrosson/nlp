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
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']

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
    """Returns a word count frequency distribution"""
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
    """Returns a trained trigram tagger

    existing : set to True if already trained tagger has been pickled
    """
    if existing:
        trigram_tagger = pickle.load(open('trained_tagger.pkl', 'rb'))
        return trigram_tagger

    # Aggregate trained sentences for N-Gram Taggers
    train_sents = nltk.corpus.brown.tagged_sents()
    train_sents += nltk.corpus.conll2000.tagged_sents()
    train_sents += nltk.corpus.treebank.tagged_sents()

    # Create instance of SubjectTrigramTagger and persist instance of it
    trigram_tagger = SubjectTrigramTagger(train_sents)
    pickle.dump(trigram_tagger, open('trained_tagger.pkl', 'wb'))

    return trigram_tagger

def tag_sentences(subject, document):
    """Returns tagged sentences using POS tagging"""
    trigram_tagger = trained_tagger(existing=True)

    # Tokenize Sentences and words
    sentences = tokenize_sentences(document)
    merge_multi_word_subject(sentences, subject)

    # Filter out sentences where subject is not present
    sentences = [sentence for sentence in sentences if subject in
                [word.lower() for word in sentence]]

    # Tag each sentence
    tagged_sents = [trigram_tagger.tag(sent) for sent in sentences]
    return tagged_sents

def merge_multi_word_subject(sentences, subject):
    """Merges multi word subjects into one single token
    ex. [('steve', 'NN', ('jobs', 'NN')] -> [('steve jobs', 'NN')]
    """
    if len(subject.split()) == 1:
        return sentences
    subject_lst = subject.split()
    sentences_lower = [[word.lower() for word in sentence]
                        for sentence in sentences]
    for i, sent in enumerate(sentences_lower):
        if subject_lst[0] in sent:
            for j, token in enumerate(sent):
                start = subject_lst[0] == token
                exists = subject_lst == sent[j:j+len(subject_lst)]
                if start and exists:
                    del sentences[i][j+1:j+len(subject_lst)]
                    sentences[i][j] = subject
    return sentences

def action_objects(sentence, subject):
    """Returns a dictionary containing:

    subject : the subject determined earlier
    action : the action verb of particular related to the subject
    object : the object the action is referring to
    phrase : list of token, tag pairs for that lie within the indexes of
                the variables above
    """
    action_objects = []
    subject_idx = next((i for i, v in enumerate(sentence)
                    if v[0].lower() == subject), None)
    for i in range(subject_idx, len(sentence)):
        data = {'subject': subject}
        found_action = False
        for j, (token, tag) in enumerate(sentence[i+1:]):
            if tag in VERBS:
                data['action'] = token
                found_action = True
            if tag in NOUNS and found_action == True:
                data['object'] = token
                data['phrase'] = sentence[i: i+j+2]
                return data
    return {}

if __name__ == '__main__':
    url = ''
    document = download_document(url)
    # document = pickle.load(open('document.pkl', 'rb'))
    document = clean_document(document)
    subject = extract_subject(document)
    print subject

    tagged_sents = tag_sentences(subject, document)

    action_objects = [action_objects(sentence, subject)
                        for sentence in tagged_sents]
    for ao in action_objects:
        if ao:
            print ao
