# import pickle
# import spacy
# import matplotlib.pyplot as plt

# pickle_abstracts = open('keywords_pickles/cs_abstracts.pickle', 'rb')
# abstracts = pickle.load(pickle_abstracts)
# pickle_abstracts.close()

# print(len(abstracts)) # 489744751

# print(len(abstracts.split('   '))) # 530825

# print(''.join(abstracts.split('   ')[:1000])) 

# abstracts = ''.join(abstracts.split('   ')[:1000])
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(abstracts)


# pickle_vocabulary = open('keywords_pickles/vocabulary.pickle', 'rb')
# vocabulary = pickle.load(pickle_vocabulary)
# pickle_vocabulary.close()
# print(len(vocabulary)) # 61275


# pickle_losses = open('keywords_pickles/losses.pickle', 'rb')
# losses = pickle.load(pickle_losses)
# pickle_losses.close()
# print(losses)
# plt.plot(losses)
# plt.xlabel('# of epochs')
# plt.ylabel('Loss')
# plt.show()

import json
import pandas as pd
import spacy
import time
import pickle
import numpy as np
from numpy.random import multinomial
from collections import Counter

# df_cs_keywords = pd.read_csv('aminer_mag_combined_cs_keywords.csv')
# cs_keywords_normalized = df_cs_keywords['NORMALIZED_NAME'][:10002].values.tolist()
# print(cs_keywords_normalized)


# 1. replace '\n' with ' '(space)
# 2. concatenate compound phrases: (Natural Language Processing)
# 3. pay attention to this dependency: 'conj'

nlp = spacy.load('en_core_web_sm')
text = 'Emotion Detection in text\ndocuments is essentially a content - based classification problem involving\nconcepts from the domains of NaturalLanguageProcessing as well as MachineLearning.' 
doc = nlp(text)

for sentence in doc.sents:
        for word in sentence:
            dep = word.dep_
            source = word.head.text
            target = word.text
            print(f' dep: {dep} source: {source} target: {target}')

