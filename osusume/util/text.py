import re
import string

import numpy as np
import pandas as pd

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class Remove:

    @staticmethod
    def numbers(text: str):
        return re.sub(r'\d+', '', text)

    @staticmethod
    def punctuations(text: str):
        return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    @staticmethod
    def whitespaces(text: str):
        return text.strip()

    @staticmethod
    def capitalfirsts(text: str):
        a_list = []
        for word in text.split():
            if word[0].isupper():
                continue
            if word[0] == '"':
                if word[1].isupper():
                    continue
            a_list.append(word)
        return ' '.join(a_list)

    @staticmethod
    def specifics(text: str, sub: str):
        return text.replace(sub, '')


class Transform:

    @staticmethod
    def stem(text: str, func: str = 'PORTER'):
        stemmer = None
        if func == 'PORTER':
            stemmer = PorterStemmer()

        if stemmer is None:
            raise OsusumeException('Stemmer function is not supported.')

        return ' '.join([stemmer.stem(y) for y in text.split()])

    @staticmethod
    def lemma(text: str, func: str = 'WORDNET'):
        lemmatizer = None
        if func == 'WORDNET':
            lemmatizer = WordNetLemmatizer()

        if lemmatizer is None:
            raise OsusumeException('Lemmatizer function is not supported.')

        return ' '.join([lemmatizer.lemmatize(y) for y in text.split()])
