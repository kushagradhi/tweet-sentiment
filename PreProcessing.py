import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class preProcessing:

    df=None
    porter=None

    def __init__(self,pdf):
        df = pdf
        df.drop(df[(df.Sentiment != 1) & (df.Sentiment != -1) & (df.Sentiment != 0)].index, inplace=True)
        df.dropna(how='any', inplace=True)

        # Remove tags
        df['Tweet'] = df['Tweet'].apply(lambda x: re.sub("<.*?>", ' ', x))

        # Remove URLs
        df['Tweet'] = df['Tweet'].apply(lambda x: re.sub("https?://[A-Za-z0-9./]+", ' ', x))

        # Remove URLs
        df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))

        # remove stopwords
        stopwords_english = stopwords.words("english")
        df['Tweet'] = df['Tweet'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stopwords_english)]))

        # lowerCase
        df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([word.lower() for word in x.split()]))
        
        #remove @mentions
        df['Tweet'] = df['Tweet'].apply(lambda x: re.sub("@.*?\s", ' ', x))

        # remove punctuation
        df['Tweet'] = df['Tweet'].apply(self.__removePunctuation)

        # stem
        df['Tweet'] = df['Tweet'].apply(self.__stem)

        self.df = df

    def __removePunctuation(self,tweet):
        return " ".join([token.strip("'\"?!,.():;#@-") for token in tweet.split()])

    def __stem(self,tweet):
        porter = PorterStemmer()
        tokens = word_tokenize(tweet)
        result = []
        for token in tokens:
            result.extend([porter.stem(token)])
        return result

