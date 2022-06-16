from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict


class NLPFeatureVectorizer(metaclass=ABCMeta):
    """Base class for NLP feature vectorization."""

    def __init__(self, corpus: list[list[str]]) -> None:
        """
        @arguments:
        corpus: list of list of string
            corpus with alldocuemnts tokenized.
        """
        self.corpus = corpus
        self.vocab = Counter()

    @abstractmethod
    def fit(self) -> None:
        self.get_vocab(self.vocab)
        pass

    @abstractmethod
    def transform(self):
        pass

    def get_vocab(self, vocab: Counter):
        """Get vocabulary"""
        for doc in self.corpus:
            self.vocab.update(doc)


class TermFrequency:
    vocab = Counter()

    def __init__(self, corpus: list[list[str]]) -> None:
        """
        @arguments:
        corpus: list of list of string
            corpus with all docuemnts tokenized.
        """
        self.corpus = corpus

    def fit(self) -> None:
        self.get_vocab(self.vocab)

        for doc in self.corpus:
            for term in doc:
                self.vocab.get(term, 0)

    def transform(self):
        pass

    def get_vocab(self, vocab: Counter):
        for doc in self.corpus:
            self.vocab.update(doc)


class DocumentFrequency(NLPFeatureVec):
    def __init__(self) -> None:
        """
        @arguments:
        corpus: list of list of string
            corpus with alldocuemnts tokenized.
        """
        self.corpus = corpus
        # self.vocab = Counter()

    normalize = lambda self, x: x.lower().strip()

    def fit(self, corpus: list[list[str]]) -> None:
        # self.get_vocab(self.vocab)
        self.df = defaultdict(int)

        for doc in self.corpus:
            doc = map(self.normalize, doc)
            unique_terms = set(doc)
            for term in unique_terms:
                self.df[term] += 1

    def transform(self):
        pass

    # get vocab seems useless, maybe I'll remove it
    # def get_vocab(self, vocab: Counter):
    #     for doc in self.corpus:
    #         doc = map(self.normalize, doc)
    #         self.vocab.update(doc)


# class InverseDocumentFrequency(...):
#     pass

# google best string to test document frequency
corpus: list[list[str]] = [
    ["The", "brown", "cat", "jumped", "the", "wall", "higher", " than", "the", "dog"],
    ["Bingo", "is", "such", "a", "nice", "dog"],
    ["The", "wages", "of", "sin", "is", "death"],
]

# tf = TermFrequency(corpus)
# tf.fit()
# print(tf.vocab)

df = DocumentFrequency()
df.fit(corpus)
# print(df.vocab)
print(df.df)
