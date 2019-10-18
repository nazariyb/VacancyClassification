import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words

from collections import Counter

nltk.download('wordnet')

__tokenizer = RegexpTokenizer(r'\w+')
__word_lem = WordNetLemmatizer()
__eng_stop_words = get_stop_words('english')


def tokenize(vacancies):
    v_tokenized = [__tokenizer.tokenize(vacancy.lower()) for vacancy in vacancies]
    v_lemmatized = [[__word_lem.lemmatize(word) for word in vacancy] for vacancy in v_tokenized]

    return v_lemmatized


def clear_and_tokenize(vacancies):
    v_tokenized = [__tokenizer.tokenize(vacancy.lower()) for vacancy in vacancies]
    v_lemmatized = [[__word_lem.lemmatize(word) for word in vacancy] for vacancy in v_tokenized]
    v_cleared = [[word for word in vacancy if word not in __eng_stop_words] for vacancy in v_lemmatized]

    return v_cleared

def get_all_words(text):
    return ' '.join([' '.join(vacancy) for vacancy in text]).split()
