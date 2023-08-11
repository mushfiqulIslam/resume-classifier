import re

import nltk

STEMMER = nltk.stem.porter.PorterStemmer()


def process_string(full_text):
    """
    Process full data of resume and remove non-english links, characters, punctuation and numbers
    """
    full_text = full_text.lower()
    full_text = re.sub('[^a-zA-Z]', ' ', full_text)
    full_text = re.sub('http\S+\s*', ' ', full_text)
    tokenized_word_list = nltk.tokenize.word_tokenize(full_text)
    tokenized_word_list = [w for w in tokenized_word_list if not w in nltk.corpus.stopwords.words('english')]
    tokenized_word_list = [STEMMER.stem(w) for w in tokenized_word_list]

    return ' '.join(tokenized_word_list)
