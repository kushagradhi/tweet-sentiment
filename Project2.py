import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import string

xls_dir = "sample.xlsx"
porter = PorterStemmer()

columns = [3,4]
df = pd.read_excel(xls_dir,sheet_name="Obama",usecols=columns,header=1,names=['Tweet','Sentiment'])

df.drop(df[(df.Sentiment==0) | (df.Sentiment==2)].index, inplace=True)
df.dropna(how='any', inplace=True)

#Remove tags
df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([re.sub('</?.>','', word) for word in x.split()]))

#Stemming
def stem(tweet):
    tokens=word_tokenize(tweet)
    result=[]
    for token in tokens:
        result.extend([porter.stem(token), " "])
    return "".join(result)
df['Tweet'] = df['Tweet'].apply(stem)

df['Tweet'].str.replace('[{}]'.format(string.punctuation), '')