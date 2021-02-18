import network
import torch
import torch.optim as optim
import time
import pickle

pickle_target_context_pairs = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/target_context_pairs.pickle', 'rb')
pickle_vocabulary = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/vocabulary.pickle', 'rb')
pickle_word_to_index = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/word_to_index.pickle', 'rb')
pickle_probabilities = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/probabilities.pickle', 'rb')

target_context_pairs = pickle.load(pickle_target_context_pairs)
vocabulary = pickle.load(pickle_vocabulary)
word_to_index = pickle.load(pickle_word_to_index)
probabilities = pickle.load(pickle_probabilities)

pickle_target_context_pairs.close()
pickle_vocabulary.close()
pickle_word_to_index.close()
pickle_probabilities.close()

negative_context_sample_size = 2
model = network.Skip_Gram_Model(embedding_size=500, vocabulary_size=len(vocabulary), negative_context_sample_size=negative_context_sample_size, probabilities=probabilities, vocabulary=vocabulary, word_to_index=word_to_index)
optimizer = optim.Adam(model.parameters())
checkpoint_save_path = '/Users/yizhuohe/Desktop/research/dbwe_spacy/checkpoints/'
start_time = time.time()
num_epoch = 10
loss_for_each_epoch = []

for epoch in range(num_epoch):
    print(f'This is epoch: {epoch}')
    train_total_loss = 0

    for i in range(len(target_context_pairs)):
        # Print as a progress bar
        if i % 500 == 0:
            print(i)
        model.zero_grad()
        target, context = target_context_pairs[i]
        loss = model(target, context)
        train_total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    loss_for_each_epoch.append(train_total_loss)

    # Save the hyperparameters of trained model
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_total_loss},
                checkpoint_save_path + 'checkpoint.tar')

end_time = time.time()
print(f'The training took: {end_time - start_time} seconds')

# Save the losses
pickle_losses = open('/Users/yizhuohe/Desktop/research/dbwe_spacy/pickles/losses.pickle', 'wb')
pickle.dump(loss_for_each_epoch, pickle_losses)
pickle_losses.close()