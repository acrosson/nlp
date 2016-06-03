import nltk
from nltk.corpus import stopwords
from gensim import corpora, models

stop = stopwords.words('english')

def run():
    """ Import the Reuters Corpus which contains 10,788 news articles"""
    from nltk.corpus import reuters
    raw_docs = [reuters.raw(fileid) for fileid in reuters.fileids()]
    print len(raw_docs)

if __name__ == '__main__':
    run()
