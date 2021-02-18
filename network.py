import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pickle
import numpy as np

class Skip_Gram_Model(nn.Module):
    def __init__(self, embedding_size, vocabulary_size, word_to_index):
        super(Skip_Gram_Model, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embeddings_target = nn.Embedding(vocabulary_size, embedding_size).to(self.device)
        self.embeddings_context = nn.Embedding(vocabulary_size, embedding_size).to(self.device)
        self.word_to_index = word_to_index
    
    def forward(self, target_word, context_word, negative_context_words):
        # Calculate the loss for this tuple based on the negative-sampling training objective function mentioned in the paper
        target_word = autograd.Variable(torch.LongTensor([self.word_to_index[target_word]])).to(self.device)
        context_word = autograd.Variable(torch.LongTensor([self.word_to_index[context_word]])).to(self.device)
        emb_target = self.embeddings_target(target_word).to(self.device)
        emb_context = self.embeddings_context(context_word).to(self.device)
        emb_product = torch.mul(emb_target, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out = torch.sum(F.logsigmoid(emb_product))

        negative_context_word = autograd.Variable(torch.LongTensor([self.word_to_index[negative_word] for negative_word in negative_context_words])).to(self.device)
        emb_negative = self.embeddings_context(negative_context_word).to(self.device)
        emb_product = torch.mul(emb_target, emb_negative)
        emb_product = torch.sum(emb_product, dim=1)
        out += torch.sum(F.logsigmoid(-emb_product))
        
        return -out # loss
