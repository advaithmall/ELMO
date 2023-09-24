import torch
import json
import csv
from tqdm import tqdm
from pprint import pprint
import numpy as np

import gensim
path = "/home/advaith/Desktop/environments/sem_5/ANLP/Project/ANLP/GoogleNews-vectors-negative300.bin"
embeddings = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import regex as re
print("Done importing...")
def preprocess(text):
    text = text.lower()
    # remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # remove numbers
    text = re.sub(r'[0-9]+', '', text)
    # remove whitespaces
    text = text.strip()
    # replace multiple white spaces with single whitespace
    text = re.sub('\s+', ' ', text)
    text = text.split()
    # replace one random word with unk
    rand_idx = np.random.randint(0, len(text))
    text[rand_idx] = "unk"
    text = " ".join(text)
    return text
    

file1 = open("data/train.csv", "r")
data = csv.reader(file1)

corp_list = []
corp_str = "sos "
next(data)
i = 0
for row in data:
    string = preprocess(row[1])
    corp_list.append(string)
    corp_str += string + " sos eos "
    i+=1
final_list = []
corp_words = corp_str.split()
# make sentences of 32 words
for i in range(0, len(corp_words), 32):
    final_list.append(corp_words[i:i+32])
print(len(final_list))


word2idx = {}
idx2word = {}
word_set = set()
for i, word in enumerate(corp_words):
    word_set.add(word)
i = 0
for word in word_set:
    if word not in word2idx.keys():
        word2idx[word] = i
        idx2word[i] = word
        i+=1

    


from torch import nn, optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Making Dataset...")
class dual_dataset(torch.utils.data.Dataset):
    def __init__(self, sent_list, word2idx, idx2word):
        self.sent_list = sent_list
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.forward_labels, self.forward_targets = self.forward()
        self.backward_labels, self.backward_targets = self.backward()
    def forward(self):
        labels = []
        targets = []
        for sent in self.sent_list:
            loc_labels = []
            loc_targets = []
            for i in range(1, len(sent)-1):
                loc_labels.append(self.word2idx[sent[i]])
                loc_targets.append(self.word2idx[sent[i+1]])
            labels.append(loc_labels)
            targets.append(loc_targets)
        return labels, targets
    def backward(self):
        labels = []
        targets = []
        for_labels = self.forward_labels
        for_targets = self.forward_targets
        for label in for_labels:
            rev_label = label[::-1]
            labels.append(rev_label)
        for target in for_targets:
            rev_target = target[::-1]
            targets.append(rev_target)
        return labels, targets
    def __len__(self):
        return len(self.sent_list)
    def __getitem__(self, idx):
        if len(self.forward_labels[idx]) != len(self.forward_labels[0]):
            idx = idx -1
        return self.forward_labels[idx], self.backward_labels[idx], self.forward_targets[idx], self.backward_targets[idx]

my_dataset = dual_dataset(final_list, word2idx, idx2word)
class lstm(nn.Module):
    def __init__(self, vocab_size):
        super(lstm, self).__init__()
        self.vocab_size = vocab_size
        # lstm with 2 stacks
        num_stacks = 2
        self.embeddings = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, 300, 1, batch_first=True)
        self.lstm1 = nn.LSTM(300, 300, 1, batch_first=True)
        self.linear = nn.Linear(300, vocab_size)
    def forward(self, x):
        embed = self.embeddings(x)
        x1, _ = self.lstm(embed)
        x2, _ = self.lstm1(x1)
        x = self.linear(x2)
        return x, (embed, x1, x2)

      


forward_model = lstm(len(word2idx)).to(device)
backward_model = lstm(len(word2idx)).to(device)
forward_model = forward_model.to(device)
backward_model = backward_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(forward_model.parameters(), lr=0.001)
optimizer2 = optim.Adam(backward_model.parameters(), lr=0.001)
dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=50, shuffle=True)
n_epochs = 25
torch.save(word2idx, "word2idx.pt")
torch.save(idx2word, "idx2word.pt")
print("Starting Training...")
for epoch in range(n_epochs):
    forward_model.train()
    backward_model.train()
    batch = 0
    for batch, (forward_labels, backward_labels, forward_targets, backward_targets) in enumerate(dataloader):
        forward_targets = torch.stack(forward_targets).to(device)
        backward_targets = torch.stack(backward_targets).to(device)
        forward_labels = torch.stack(forward_labels).to(device)
        backward_labels = torch.stack(backward_labels).to(device)
        # change the shape of forward_vcts and backward_vcts to batch-size, seq_len, embedding_dim
        forward_targets = forward_targets.view(forward_targets.shape[1], forward_targets.shape[0])
        backward_targets = backward_targets.view(backward_targets.shape[1], backward_targets.shape[0])
        forward_labels = forward_labels.view(forward_labels.shape[1], forward_labels.shape[0])
        backward_labels = backward_labels.view(backward_labels.shape[1], backward_labels.shape[0])
        # forward pass
        optimizer1.zero_grad()
        forward_out,_  = forward_model(forward_labels)
        forward_out = forward_out.view(forward_out.shape[0]*forward_out.shape[1], forward_out.shape[2])
        forward_targets = forward_targets.view(forward_targets.shape[0]*forward_targets.shape[1])
        forward_loss = loss_fn(forward_out, forward_targets)
        forward_loss.backward()
        optimizer1.step()
        # backward pass
        optimizer2.zero_grad()
        backward_out,_ = backward_model(backward_labels)
        backward_out = backward_out.view(backward_out.shape[0]*backward_out.shape[1], backward_out.shape[2])
        backward_targets = backward_targets.view(backward_targets.shape[0]*backward_targets.shape[1])
        backward_loss = loss_fn(backward_out, backward_targets)
        backward_loss.backward()
        optimizer2.step()
        forward_peplexity = torch.exp(forward_loss)
        backward_peplexity = torch.exp(backward_loss)
        print("Epoch: {}, Batch: {}, Forward Loss: {}, Backward Loss: {}, Forward Peplexity: {}, Backward Peplexity: {}".format(epoch, batch, forward_loss, backward_loss, forward_peplexity, backward_peplexity))
    torch.save(forward_model, "forward_model.pt")
    torch.save(backward_model, "backward_model.pt")

        

   
