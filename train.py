import network
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import numpy as np
from numpy.random import multinomial

pickle_target_context_pairs = open('keywords_pickles/target_context_pairs.pickle', 'rb')
pickle_vocabulary = open('keywords_pickles/vocabulary.pickle', 'rb')
pickle_word_to_index = open('keywords_pickles/word_to_index.pickle', 'rb')
pickle_probabilities = open('keywords_pickles/probabilities.pickle', 'rb')

target_context_pairs = pickle.load(pickle_target_context_pairs)
vocabulary = pickle.load(pickle_vocabulary)
word_to_index = pickle.load(pickle_word_to_index)
probabilities = pickle.load(pickle_probabilities)

pickle_target_context_pairs.close()
pickle_vocabulary.close()
pickle_word_to_index.close()
pickle_probabilities.close()

negative_context_sample_size = 5

model = network.Skip_Gram_Model(embedding_size=128, vocabulary_size=len(vocabulary), word_to_index=word_to_index)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters())

checkpoint_save_path = 'keywords_checkpoints/'
start_time = time.time()
num_epoch = 15
loss_for_each_epoch = []

# Generate the negative context words for negative-sampling
def generate_negative_context_words(sample_size, probabilities, vocabulary):
  negative_context_words = []
  sample_result = np.array(multinomial(sample_size, list(probabilities.values())))
  for word_index, word_count in enumerate(sample_result):
    for _ in range(word_count):
      negative_context_words.append(vocabulary[word_index])
  return negative_context_words

pickle_target_context_pairs = open('keywords_pickles/target_context_pairs.pickle', 'rb')
target_context_pairs = pickle.load(pickle_target_context_pairs)
pickle_target_context_pairs.close()

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    print(f'This is epoch: {epoch}')
    train_total_loss = 0

    for i in range(len(target_context_pairs)):
        # Print as a progress bar
        if i % 5000 == 0:
            print(i)
            # print(f'time spent: {time.time() - epoch_start_time}')
        optimizer.zero_grad()
        target, context = target_context_pairs[i]
        negative_context_words = generate_negative_context_words(negative_context_sample_size, probabilities, vocabulary)
        loss = model(target, context, negative_context_words)
        train_total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    loss_for_each_epoch.append(train_total_loss)

    print(f'This epoch took: {time.time() - epoch_start_time} seconds')

    # Save the losses
    pickle_losses = open('keywords_pickles/losses.pickle', 'wb')
    pickle.dump(loss_for_each_epoch, pickle_losses)
    pickle_losses.close()

    # Save the hyperparameters of trained model
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_total': train_total_loss},
                checkpoint_save_path + f'checkpoint_epoch_{epoch}.tar')

end_time = time.time()
print(f'The training took: {end_time - start_time} seconds')
