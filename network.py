import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pickle
import numpy as np
from numpy.random import multinomial

# Generate the negative context words for negative-sampling
def generate_negative_context_words(sample_size, probabilities, vocabulary):
  negative_context_words = []
  sample_result = np.array(multinomial(sample_size, list(probabilities.values())))
  for word_index, word_count in enumerate(sample_result):
    for _ in range(word_count):
      negative_context_words.append(vocabulary[word_index])
  return negative_context_words

class Skip_Gram_Model(nn.Module):
    def __init__(self, embedding_size, vocabulary_size, negative_context_sample_size, probabilities, vocabulary, word_to_index):
        super(Skip_Gram_Model, self).__init__()
        self.embeddings_target = nn.Embedding(vocabulary_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocabulary_size, embedding_size)
        self.negative_context_sample_size = negative_context_sample_size
        self.probabilities = probabilities
        self.vocabulary = vocabulary
        self.word_to_index = word_to_index
    
    def forward(self, target_word, context_word):
        # Calculate the loss for this tuple based on the negative-sampling training objective function mentioned in the paper
        target_word = autograd.Variable(torch.LongTensor([self.word_to_index[target_word]]))
        context_word = autograd.Variable(torch.LongTensor([self.word_to_index[context_word]]))
        emb_target = self.embeddings_target(target_word)
        emb_context = self.embeddings_context(context_word)
        emb_product = torch.mul(emb_target, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out = torch.sum(F.logsigmoid(emb_product))

        negative_context_words = generate_negative_context_words(self.negative_context_sample_size, self.probabilities, self.vocabulary)
        for i in range(len(negative_context_words)):
            negative_context_word = autograd.Variable(torch.LongTensor([self.word_to_index[negative_context_words[i]]]))
            emb_negative = self.embeddings_context(negative_context_word)
            emb_product = torch.mul(emb_target, emb_negative)
            emb_product = torch.sum(emb_product, dim=1)
            out += torch.sum(F.logsigmoid(-emb_product))
        
        return -out # loss
