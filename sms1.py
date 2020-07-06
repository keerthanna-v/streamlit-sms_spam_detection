# importing packages

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from text_processing import puncation,stopwordsremoval,lemt
from sklearn.metrics import accuracy_score
import pickle
# loading the dataset
data=pd.read_csv(r"C:\Users\keerthanna\Documents\spam.csv")
data.rename(columns={
   'EmailText':'text',
    'Label':'type'
},inplace=True)
# data processing

# removal of puncations

data['text']=data['text'].apply(puncation)


# removal of stopwords
data['text']=data['text'].apply(lambda x: x.lower())

data['text']=data['text'].apply(stopwordsremoval)
data['text']=data['text'].apply(lemt)

x_train,x_test,y_train,y_test=train_test_split(data['text'], data['type'], test_size=0.3,random_state=1)

tfidf_vectorizer= TfidfVectorizer()

tfidf_train=tfidf_vectorizer.fit_transform(x_train)

tfidf_test=tfidf_vectorizer.transform(x_test)
def test(text):
    return tfidf_vectorizer.transform([text])

clf=MultinomialNB()

clf.fit(tfidf_train,y_train)


pickle.dump(clf,open("model1.pkl",'wb'))
model=pickle.load(open('model1.pkl','rb'))