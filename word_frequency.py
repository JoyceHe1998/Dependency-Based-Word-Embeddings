import pickle

pickle_counts = open('keywords_pickles/word_counts.pickle', 'rb')

word_counts = pickle.load(pickle_counts)

pickle_counts.close()
# print(word_counts)
a = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1])}
print(a)