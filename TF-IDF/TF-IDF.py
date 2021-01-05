# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 05:44:59 2021

@author: sanjay
"""

import nltk

paragraph = """In 3000 years of our history, people from all over the worlhave 
come and invaded us, captured our lands, conquered our minds. 
From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, 
the British, the French, the Dutch, all of them came and looted us, 
took over what was ours. Yet, we have not done this to any other nation. 
We have not conquered anyone. We have not grabbed their land, their 
culture, their history and tried to enforce our way of life on them. 
Why? Because we respect the freedom of others."""

#cleaning the text

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


Lr = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [Lr.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#creating TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()