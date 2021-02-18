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
excluded_dependencies = ['ROOT', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'det', 'disclosure', 'expl', 'mark', 'neg', 'pcomp', 'poss', 'preconj', 'punct', 'predet', 'ref']

# Generate target_context tuples 
def generate_target_context_tuples(text):
    doc = nlp(text)
    start_time = time.time()
    all_words = [] # Used to get word_counts and vocabulary
    target_context_tuples = []
    for sentence in doc.sents:
        # The following 2 dictionaroes are used to collapse prepositional relations
        prep_dict = {}
        pobj_dict = {}
        for word in sentence:
            dep = word.dep_

            if not dep in excluded_dependencies: # Only add the target-context pair that has meaningful dependencies
                # Convert the target and context words to lower case
                source = word.head.text.lower()
                target = word.text.lower()

                if source.isalpha() and target.isalpha(): 
                    all_words.append(source) 
                    all_words.append(target) 
                    if dep == 'prep':
                        prep_dict[word.i+1] = source
                    elif dep == 'pobj':
                        pobj_dict[word.head.i+1] = target
                    else:
                        target_context_tuples.append((source, target))
                        target_context_tuples.append((target, source))

        # Add all collapsed prepositional relations        
        for i in prep_dict.keys():
            if i in pobj_dict:
                target_context_tuples.append((prep_dict[i], pobj_dict[i]))
                target_context_tuples.append((pobj_dict[i], prep_dict[i]))
    print(f'Parsing the partial corpus took: {time.time() - start_time} seconds')
    print(f'There are {len(target_context_tuples)} target_context pairs')
    return (target_context_tuples, all_words)

# need to split the twenty_train into smaller partial corpus becasue the spacy parser has limit on character length of text
pickle_abstracts = open('keywords_pickles/cs_abstracts.pickle', 'rb')
abstracts = pickle.load(pickle_abstracts)
pickle_abstracts.close()

start_time = time.time()
target_context_tuples = []
all_words = []

for i in range(9): # 10000 / 530825
    partial_corpus = ' '.join(abstracts.split('   ')[i*1000 : (i+1)*1000])
    target_context_tuples_partial, all_words_partial = generate_target_context_tuples(partial_corpus)
    target_context_tuples += target_context_tuples_partial
    all_words += all_words_partial

    # save parsed result
    pickle_target_context_pairs = open('keywords_pickles/target_context_pairs.pickle', 'wb')
    pickle_all_words = open('keywords_pickles/all_words.pickle', 'wb')
    pickle.dump(target_context_tuples, pickle_target_context_pairs)
    pickle.dump(all_words, pickle_all_words)
    pickle_target_context_pairs.close()
    pickle_all_words.close()

    # print as a progress bar
    print(i)

##########################
##########################

pickle_target_context_pairs = open('keywords_pickles/target_context_pairs.pickle', 'rb')
pickle_all_words = open('keywords_pickles/all_words.pickle', 'rb')
target_context_tuples = pickle.load(pickle_target_context_pairs)
all_words = pickle.load(pickle_all_words)
pickle_target_context_pairs.close()
pickle_all_words.close()

print(f'total number of target_context_tuples: {len(target_context_tuples)}') # 1044726
print(f'total number of all_words: {len(all_words)}') # 1332670            

###########
word_counts = dict(Counter(all_words))
vocabulary = list(word_counts.keys()) # vocabulary = list(dict.fromkeys(all_words))
word_to_index = {w : idx for (idx, w) in enumerate(vocabulary)}
index_to_word = {idx : w for (idx, w) in enumerate(vocabulary)}

# Generate probabilities array for negative-sampling
denominator = sum([word_count**0.75 for word_count in word_counts.values()])
probabilities = {}
for word in word_counts:
    probabilities[word] = word_counts[word]**0.75 / denominator

pickle_vocabulary = open('keywords_pickles/vocabulary.pickle', 'wb')
pickle_word_counts = open('keywords_pickles/word_counts.pickle', 'wb')
pickle_word_to_index = open('keywords_pickles/word_to_index.pickle', 'wb')
pickle_index_to_word = open('keywords_pickles/index_to_word.pickle', 'wb')
pickle_probabilities = open('keywords_pickles/probabilities.pickle', 'wb')

pickle.dump(vocabulary, pickle_vocabulary)
pickle.dump(word_counts, pickle_word_counts)
pickle.dump(word_to_index, pickle_word_to_index)
pickle.dump(index_to_word, pickle_index_to_word)
pickle.dump(probabilities, pickle_probabilities)

pickle_vocabulary.close()
pickle_word_counts.close()
pickle_word_to_index.close()
pickle_index_to_word.close()
pickle_probabilities.close()

print('All work finished!!!!')