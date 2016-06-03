import re
import random
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models

stop = stopwords.words('english')
add_stopwords = ['said', 'mln', 'billion', 'million', 'pct', 'would', 'inc', 'company', 'corp']
stop += add_stopwords

def ie_preprocess(document):
    document = re.sub('[^A-Za-z ]+', '', document)
    document = ' '.join([i for i in document.lower().split()
                        if i not in stop])
    document = nltk.word_tokenize(document)
    return document

def remove_infrequent_words(docs):

    """Remove all the words that only occur once"""

    from collections import defaultdict
    frequency = defaultdict(int)
    for doc in docs:
        for token in doc:
            frequency[token] += 1

    docs = [[token for token in doc if frequency[token] > 1]
             for doc in docs]
    return docs

def run():

    """Import the Reuters Corpus which contains 10,788 news articles"""

    from nltk.corpus import reuters
    raw_docs = [reuters.raw(fileid) for fileid in reuters.fileids()]

    # Select 100 documents randomly
    rand_idx = random.sample(range(len(raw_docs)), 100)
    raw_docs = [raw_docs[i] for i in rand_idx]

    # Preprocess Documents
    tokenized_docs = [ie_preprocess(doc) for doc in raw_docs]

    # Remove single occurance words
    docs = remove_infrequent_words(tokenized_docs)

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # Build LDA model
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=10)
    for topic in lda.show_topics():
        print topic


if __name__ == '__main__':
    run()
