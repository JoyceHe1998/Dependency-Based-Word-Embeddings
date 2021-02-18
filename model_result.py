import torch
import network
import pickle

checkpoint = torch.load('/Users/yizhuohe/Desktop/research/dbwe_spacy/checkpoints/' + 'checkpoint.tar')

pickle_target_context_pairs = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/target_context_pairs.pickle', 'rb')
pickle_vocabulary = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/vocabulary.pickle', 'rb')
pickle_word_to_index = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/word_to_index.pickle', 'rb')
pickle_index_to_word = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/index_to_word.pickle', 'rb')
pickle_probabilities = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/probabilities.pickle', 'rb')

target_context_pairs = pickle.load(pickle_target_context_pairs)
vocabulary = pickle.load(pickle_vocabulary)
word_to_index = pickle.load(pickle_word_to_index)
index_to_word = pickle.load(pickle_index_to_word)
probabilities = pickle.load(pickle_probabilities)

# print(vocabulary)

pickle_target_context_pairs.close()
pickle_vocabulary.close()
pickle_word_to_index.close()
pickle_index_to_word.close()
pickle_probabilities.close()

model = network.Skip_Gram_Model(embedding_size=500, vocabulary_size=len(vocabulary), negative_context_sample_size=2, probabilities=probabilities, vocabulary=vocabulary, word_to_index=word_to_index)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def get_most_similar_words(word, topn=10):
    word_distance = []
    trained_model_embedding = model.embeddings_target
    i = word_to_index[word]
    lookup_tensor_i = torch.tensor([i], dtype=torch.long)
    v_i = trained_model_embedding(lookup_tensor_i)
    for j in range(len(vocabulary)):
        if j != i:
            tensor_j = torch.tensor([j], dtype=torch.long)
            v_j = trained_model_embedding(tensor_j)
            word_distance.append((index_to_word[j], cos_sim(v_i, v_j)))
    word_distance.sort(key=lambda x: x[1])
    return word_distance[:topn]

most_similar_words = get_most_similar_words('chinese')
print(f'the most similar words are: {most_similar_words}')
