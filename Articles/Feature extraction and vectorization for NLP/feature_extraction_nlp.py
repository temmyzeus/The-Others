from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict

import numpy as np


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


class TermFrequencyVectorizer(NLPFeatureVectorizer):
    vocab = Counter()

    def __init__(self, corpus: list[list[str]]) -> None:
        """
        Term Frequency = Number of times T appearts in document / Number of terms in document

        @arguments
        corpus: list of list of string
            corpus with all docuemnts tokenized.
        """
        self.corpus = corpus

    def fit(self) -> None:
        self.get_vocab(self.vocab)
        terms = list(self.vocab.keys())

        n_docs = len(self.corpus)
        n_terms = len(self.vocab)
        tf_vector = np.zeros((n_docs, n_terms))

        for doc_index, doc in enumerate(self.corpus):
            n_terms_in_doc = len(doc)
            unique_terms = set(doc)

            for term in unique_terms:
                term_pos = terms.index(term)
                self.vocab.get(term, 0)
                tf_vector[doc_index, term_pos] = doc.count(term) / n_terms_in_doc

        return tf_vector

    def transform(self):
        pass


class DocumentFrequency(NLPFeatureVectorizer):
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

corpus2 = [
    [
        "New",
        "York",
        "(",
        "CNN",
        "Business",
        ")",
        "The",
        "Dow",
        "fell",
        "460",
        "points",
        "Friday",
        "after",
        "a",
        "US",
        "recession",
        "indicator",
        "blinked",
        "red",
        "and",
        "a",
        "report",
        "on",
        "German",
        "manufacturing",
        "raised",
        "concerns",
        "about",
        "Europe",
        "'",
        "s",
        "most",
        "important",
        "economy",
        ".",
    ],
    [
        "The",
        "index",
        "shed",
        "1",
        ".",
        "8",
        "%,",
        "while",
        "the",
        "S",
        "&",
        "P",
        "500",
        "closed",
        "down",
        "1",
        ".",
        "9",
        "%.",
        "The",
        "Nasdaq",
        "plunged",
        "2",
        ".",
        "5",
        "%.",
        "It",
        "was",
        "the",
        "worst",
        "performance",
        "for",
        "all",
        "three",
        "major",
        "indexes",
        "since",
        "January",
        "3",
        ".",
    ],
    [
        "The",
        "yield",
        "on",
        "3",
        "-",
        "month",
        "Treasuries",
        "rose",
        "above",
        "the",
        "rate",
        "on",
        "10",
        "-",
        "year",
        "Treasuries",
        "for",
        "the",
        "first",
        "time",
        "since",
        "2007",
        "—",
        "a",
        "shift",
        "that",
        "scared",
        "Wall",
        "Street",
        ".",
        "Investors",
        "have",
        "piled",
        "back",
        "into",
        "stocks",
        "after",
        "a",
        "sell",
        "-",
        "off",
        "in",
        "late",
        "2018",
        ".",
    ],
    [
        "The",
        "flattening",
        "yield",
        "curve",
        ",",
        "or",
        "the",
        "difference",
        "between",
        "short",
        "-",
        "and",
        "long",
        "-",
        "term",
        "rates",
        ",",
        "has",
        "worried",
        "investors",
        "for",
        "months",
        ".",
        "A",
        "narrowing",
        "spread",
        "is",
        "typically",
        "seen",
        "as",
        "sign",
        "that",
        "long",
        "-",
        "term",
        "economic",
        "confidence",
        "is",
        "dwindling",
        ".",
        "For",
        "decades",
        ",",
        "an",
        "inversion",
        "has",
        "been",
        "a",
        "reliable",
        "predictor",
        "of",
        "a",
        "future",
        "recession",
        ".",
    ],
    [
        "Friday",
        "'",
        "s",
        "flip",
        "added",
        "to",
        "pressure",
        "on",
        "the",
        "Dow",
        "that",
        "was",
        "building",
        "before",
        "US",
        "markets",
        "opened",
        ".",
    ],
    [
        "The",
        "index",
        "stumbled",
        "at",
        "the",
        "bell",
        "on",
        "poor",
        "manufacturing",
        "data",
        "from",
        "Germany",
        ",",
        "which",
        "also",
        "spelled",
        "trouble",
        "for",
        "the",
        "country",
        "'",
        "s",
        "bond",
        "market",
        ".",
        "The",
        "yield",
        "on",
        "Germany",
        "'",
        "s",
        "benchmark",
        "10",
        "-",
        "year",
        "government",
        "bond",
        "fell",
        "below",
        "zero",
        "for",
        "the",
        "first",
        "time",
        "since",
        "October",
        "2016",
        ".",
    ],
    [
        "That",
        "news",
        "out",
        "of",
        "Europe",
        "fueled",
        "Wall",
        "Street",
        "'",
        "s",
        "ongoing",
        "concerns",
        "about",
        "slowing",
        "global",
        "growth",
        ".",
    ],
    [
        "Investors",
        "remain",
        "jittery",
        "about",
        "Brexit",
        "and",
        "the",
        "lasting",
        "effects",
        "of",
        "the",
        "trade",
        "fight",
        "between",
        "the",
        "United",
        "States",
        "and",
        "China",
        ",",
        "even",
        "as",
        "Washington",
        "and",
        "Beijing",
        "move",
        "toward",
        "a",
        "deal",
        ".",
    ],
    [
        "And",
        "they",
        "are",
        "unsure",
        "how",
        "to",
        "interpret",
        "the",
        "Federal",
        "Reserve",
        "'",
        "s",
        "signal",
        "that",
        "it",
        "won",
        "'",
        "t",
        "hike",
        "interest",
        "rates",
        "this",
        "year",
        ".",
        "On",
        "one",
        "hand",
        ",",
        "maintaining",
        "rates",
        "could",
        "ensure",
        "that",
        "credit",
        "keeps",
        "flowing",
        "and",
        "the",
        "10",
        "-",
        "year",
        "bull",
        "market",
        "continues",
        ".",
        "But",
        "it",
        "also",
        "speaks",
        "to",
        "concern",
        "about",
        "the",
        "country",
        "'",
        "s",
        "economic",
        "health",
        ",",
        "which",
        "could",
        "stifle",
        "investment",
        ".",
    ],
    [
        "For",
        "the",
        "week",
        ",",
        "the",
        "Dow",
        ",",
        "S",
        "&",
        "P",
        "500",
        "and",
        "Nasdaq",
        "finished",
        "modestly",
        "lower",
        ".",
    ],
    [
        "But",
        "bank",
        "stocks",
        ",",
        "which",
        "are",
        "particularly",
        "sensitive",
        "to",
        "interest",
        "rates",
        "and",
        "economic",
        "worries",
        ",",
        "took",
        "a",
        "beating",
        ".",
        "The",
        "KBW",
        "Bank",
        "index",
        "(",
        "BKX",
        ")",
        "dropped",
        "more",
        "than",
        "8",
        "%",
        "in",
        "the",
        "past",
        "week",
        ".",
    ],
    [
        "White",
        "House",
        "economic",
        "adviser",
        "Larry",
        "Kudlow",
        "told",
        "CNBC",
        "last",
        "year",
        "that",
        "the",
        "spread",
        "between",
        "3",
        "-",
        "month",
        "and",
        "10",
        "-",
        "year",
        "Treasury",
        "yields",
        "was",
        "important",
        "to",
        "watch",
        ".",
    ],
    [
        '"',
        "It",
        "'",
        "s",
        "actually",
        "not",
        "10s",
        "to",
        "2s",
        ";",
        "it",
        "'",
        "s",
        "10s",
        "to",
        "3",
        "-",
        "month",
        "Treasury",
        "bills",
        ',"',
        "Kudlow",
        "said",
        "last",
        "May",
        ".",
        "He",
        "was",
        "referring",
        "to",
        "the",
        "spread",
        "between",
        "2",
        "-",
        "month",
        "and",
        "10",
        "-",
        "year",
        "Treasury",
        "yields",
        ",",
        "which",
        "is",
        "also",
        "closely",
        "monitored",
        ".",
    ],
    [
        "Michael",
        "Darda",
        ",",
        "chief",
        "economist",
        "and",
        "market",
        "strategist",
        "at",
        "MKM",
        "Partners",
        ",",
        "said",
        "in",
        "a",
        "note",
        "that",
        "investors",
        "should",
        "wait",
        "for",
        "weekly",
        "and",
        "monthly",
        "averages",
        "to",
        "show",
        "an",
        "inversion",
        "before",
        "they",
        "read",
        "it",
        "as",
        "a",
        '"',
        "powerful",
        "recession",
        "signal",
        '."',
    ],
    [
        "And",
        "he",
        "noted",
        "that",
        "on",
        "average",
        ",",
        "recessions",
        "occur",
        "12",
        "months",
        "after",
        "an",
        "inversion",
        "—",
        "not",
        "immediately",
        ".",
    ],
]

tf_vectorizer = TermFrequencyVectorizer(corpus)
tf_vectorizer2 = TermFrequencyVectorizer(corpus2)
tf = tf_vectorizer.fit()
tf2 = tf_vectorizer2.fit()


# df = DocumentFrequency()
# df.fit(corpus)
# # print(df.vocab)
# print(df.df)
