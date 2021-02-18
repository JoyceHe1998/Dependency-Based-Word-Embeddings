import json
import pandas as pd
import spacy
import pickle
import time

# total: 1796000 papers (all types)
pickle_abstracts = open('keywords_pickles/cs_abstracts.pickle', 'wb')

i = 0
abstracts = ''
nlp = spacy.load('en_core_web_sm')

with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
    start_time = time.time()
    for line in f:
        if i % 1000 == 0:
            print(f'i: {i}')
        json_object = json.loads(line)
        if 'cs.' in json_object['categories']:
            i += 1
            abstract = json_object['abstract'].replace('\n', ' ')
            # merge compound words
            doc = nlp(abstract)
            compound_pairs = []
            for sentence in doc.sents:
                for word in sentence:
                    dep = word.dep_
                    if dep == 'compound':
                        first = word.text
                        second = word.head.text
                        compound_pairs.append((first, second))
            for (first, second) in compound_pairs:
                abstract = abstract.replace(first + ' ' + second, first + second)         
            abstracts += abstract
    print(f'time spent: {time.time() - start_time}') # time spent: 10994.584422111511
    
pickle.dump(abstracts, pickle_abstracts)
pickle_abstracts.close()
print(i) # 480326