import re
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

## Number of characters
def count_chars(text):
    return len(text)

## Number of words
def count_words(text):
    return len(text.split())

## Number of capital characters
def count_capital_chars(text):
    count = 0
    for i in text:
        if i.isupper():
            count += 1
    return count

## Number of capital words
def count_capital_words(text):
    return sum(map(str.isupper,text.split()))

## Number of sentences
def count_sent(text):
    return len(sent_tokenize(text))

## Number of unique words
def count_unique_words(text):
    return len(set(text.split()))

## Number of stopwords
def count_stopwords(text):
    stop_words = set(stopwords.words('english'))  
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)

## Number of hashtags
def count_hashtags(text):
    x = re.findall(r'(#w[A-Za-z0-9]*)', text)
    return len(x) 