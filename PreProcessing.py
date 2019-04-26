import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from Utils import translator
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# SelectKBest
class preProcessing:

    df=None
    porter=None

    def __init__(self, pdf):
        df = pdf
        df.drop(df[(df.Sentiment != 1) & (df.Sentiment != -1) & (df.Sentiment != 0)].index, inplace=True)
        df.dropna(how='any', inplace=True)

        df['Tweet'] = df['Tweet'].apply(lambda tweet: tweet.lower())
        df['Tweet'] = df['Tweet'].apply(lambda tweet: tweet.replace('\n', '').strip())
        df['Tweet'] = df['Tweet'].apply(lambda tweet: tweet.replace(u'\u2018', "'").replace(u'\u2019', "'"))
        df['Tweet'] = df['Tweet'].apply(lambda tweet: tweet.replace('n\'t', ' not'))
        df['Tweet'] = df['Tweet'].apply(lambda tweet: re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet))
        df['Tweet'] = df['Tweet'].apply(lambda tweet: re.sub(r'[^\x00-\x7F]+','', tweet))
        df['Tweet'] = df['Tweet'].apply(lambda tweet: tweet.replace('RT',''))
        df['Tweet'] = df['Tweet'].apply(lambda tweet: re.sub('@[^\s]+', 'AT_USER', tweet))
        df['Tweet'] = df['Tweet'].apply(lambda tweet: re.sub("<.*?>", ' ', tweet))
        df['Tweet'] = df['Tweet'].apply(lambda tweet: tweet.replace('.', ''))

        punctuations = list(string.punctuation) + ['``', "''","'"]
        df['Tweet'] = df['Tweet'].apply(lambda tweet:
                        " ".join([word for word in word_tokenize(tweet) if word not in punctuations]))

        df['Tweet'] = df['Tweet'].apply(lambda tweet: tweet.replace('\'', ''))

        useless = ['gop', 'debate',
                       'gopdeb', 'gopdebate','gopdebates', 'fox','cnn',"'s", 'news','foxnew', 'foxnews', 'amp','a','an','the']

        df['Tweet'] = df['Tweet'].apply(lambda tweet:
                       " ".join([word for word in word_tokenize(tweet) if word not in useless]))
        x = [tweet for tweet in df['Tweet']]
        df['Tweet'] = df['Tweet'].apply(self.__stem)
        df['Tweet'] = df['Tweet'].apply(lambda tweet: [word for word in tweet if len(word)>1])
        y = [" ".join(tweet) for tweet in df['Tweet']]

        all_words = [word for tweet in df['Tweet'] for word in tweet]
        word_counter = Counter(all_words)
        most_common_words = word_counter.most_common(100)
        #print(most_common_words)

        #for i in range(len(x)):
        #    print (x[i],' ----- ',y[i],'\n')

        #exit()
        self.df = df

    # def __init__(self,pdf):
    #     df = pdf
    #     df.drop(df[(df.Sentiment != 1) & (df.Sentiment != -1) & (df.Sentiment != 0)].index, inplace=True)
    #     df.dropna(how='any', inplace=True)
    #
    #     # Remove tags
    #     df['Tweet'] = df['Tweet'].apply(lambda x: re.sub("<.*?>", ' ', x))
    #
    #     # Remove URLs
    #     df['Tweet'] = df['Tweet'].apply(lambda x: re.sub("https?://[A-Za-z0-9./]+", ' ', x))
    #
    #     df['Tweet'] = df['Tweet'].apply(lambda x: re.sub("[0-9]*", '', x))
    #
    #     # Remove URLs
    #     #df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    #
    #     # remove stopwords
    #     stopwords_english = stopwords.words("english")
    #     df['Tweet'] = df['Tweet'].apply(
    #         lambda x: ' '.join([word for word in x.split() if word not in (stopwords_english)]))
    #
    #     # lowerCase
    #     df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([word.lower() for word in x.split()]))
    #
    #     #remove @mentions
    #     #df['Tweet'] = df['Tweet'].apply(lambda x: re.sub("@.*?\s", ' ', x))
    #
    #     # remove punctuation
    #     df['Tweet'] = df['Tweet'].apply(self.__removePunctuation)
    #
    #
    #     # stem
    #     df['Tweet'] = df['Tweet'].apply(self.__stem)
    #
    #     #[print(tweet) for tweet in df['Tweet']]
    #
    #     self.df = df

    def __preprocessTweet(self,tweet):
        tweet.lower()
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
        tweet = re.sub('[\s]+', ' ', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        return tweet.split()

    def __removePunctuation(self,tweet):
        return " ".join([token.strip(string.punctuation) for token in tweet.split()])

    def __stem(self,tweet):
        porter = PorterStemmer()
        tokens = word_tokenize(tweet)
        result = []
        for token in tokens:
            result.extend([porter.stem(token)])
        return result

    def __dataAnalysis(self,pdf):
        df = pdf
        df.drop(df[(df.Sentiment != 1) & (df.Sentiment != -1) & (df.Sentiment != 0)].index, inplace=True)
        df.dropna(how='any', inplace=True)
        df.head()

        print('Dataset size:', df.shape)
        print('Columns are:', df.columns)
        df.info()

        sns.countplot(x='Sentiment', data=df)
        plt.show()
        exit()