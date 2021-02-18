import time
import spacy
import pickle
import numpy as np
from numpy.random import multinomial
from collections import Counter
import pandas as pd

df_ted = pd.read_csv('/Users/yizhuohe/Desktop/transcripts.csv')
transcripts = df_ted['transcript']

# Use spacy to replace CoreNLPClient since the latter has limit on the size of corpus and is slow
nlp = spacy.load('en_core_web_sm')

# Exclude the following dependencies
excluded_dependencies = ['ROOT', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'det', 'disclosure', 'expl', 'mark', 'neg', 'pcomp', 'poss', 'preconj', 'punct', 'predet', 'ref']

# text = 'Australian scientist discovers star with telescope.'
# text = 'Australian scientist discovers star with telescope. I sat on the chair. I like scientist.'
# text = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/alice_in_wonderland_half.txt').read()
text = transcripts[353] 
doc = nlp(text)
# doc = nlp(text.decode('utf8'))

# Generate target_context tuples 
def generate_target_context_tuples():
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
    print(f'Parsing the entire corpus took: {time.time() - start_time} seconds')
    print(f'There are {len(target_context_tuples)} target_context pairs')
    return (target_context_tuples, all_words)

target_context_tuples, all_words = generate_target_context_tuples()
word_counts = dict(Counter(all_words))
vocabulary = list(word_counts.keys()) # vocabulary = list(dict.fromkeys(all_words))
word_to_index = {w : idx for (idx, w) in enumerate(vocabulary)}
index_to_word = {idx : w for (idx, w) in enumerate(vocabulary)}

# Generate probabilities array for negative-sampling
denominator = sum([word_count**0.75 for word_count in word_counts.values()])
probabilities = {}
for word in word_counts:
    probabilities[word] = word_counts[word]**0.75 / denominator

# print(f'words: {words}')
# print(f'vocabulary: {vocabulary}')
# print(f'word_counts: {word_counts}')

# Save the parsed data
# pickle_doc = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/doc.pickle', 'wb')
pickle_target_context_pairs = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/target_context_pairs.pickle', 'wb')
pickle_all_words = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/all_words.pickle', 'wb')
pickle_vocabulary = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/vocabulary.pickle', 'wb')
pickle_word_counts = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/word_counts.pickle', 'wb')
pickle_word_to_index = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/word_to_index.pickle', 'wb')
pickle_index_to_word = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/index_to_word.pickle', 'wb')
pickle_probabilities = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/probabilities.pickle', 'wb')

pickle.dump(target_context_tuples, pickle_target_context_pairs)
pickle.dump(all_words, pickle_all_words)
pickle.dump(vocabulary, pickle_vocabulary)
pickle.dump(word_counts, pickle_word_counts)
pickle.dump(word_to_index, pickle_word_to_index)
pickle.dump(index_to_word, pickle_index_to_word)
pickle.dump(probabilities, pickle_probabilities)

pickle_target_context_pairs.close()
pickle_all_words.close()
pickle_vocabulary.close()
pickle_word_counts.close()
pickle_word_to_index.close()
pickle_index_to_word.close()
pickle_probabilities.close()
