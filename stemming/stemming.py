# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 04:19:35 2021

@author: sanjay
"""
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """In 3000 years of our history, people from all over the worlhave come and invaded us, captured our lands, conquered our minds. 
From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, 
the British, the French, the Dutch, all of them came and looted us, 
took over what was ours. Yet, we have not done this to any other nation. 
We have not conquered anyone. We have not grabbed their land, their 
culture, their history and tried to enforce our way of life on them. 
Why? Because we respect the freedom of others."""

sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()
#stemming

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)