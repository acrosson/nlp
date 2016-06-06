from nltk import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger

class SubjectTrigramTagger(object):

    """ Creates an instance of NLTKs TrigramTagger with a backoff
    tagger of a bigram tagger a unigram tagger and a default tagger that sets
    all words to nouns (NN)
    """

    def __init__(self, train_sents):

        """
        train_sents: trained sentences which have already been tagged.
                Currently using Brown, conll2000, and TreeBank corpuses
        """

        t0 = DefaultTagger('NN')
        t1 = UnigramTagger(train_sents, backoff=t0)
        t2 = BigramTagger(train_sents, backoff=t1)
        self.tagger = TrigramTagger(train_sents, backoff=t2)

    def tag(self, tokens):
        return self.tagger.tag(tokens)
