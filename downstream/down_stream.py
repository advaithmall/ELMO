import torch
import json
import csv
from tqdm import tqdm
from pprint import pprint
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
import gensim
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch import nn, optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    return text

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

def make_dataset(filename, word2idx, idx2word):
    filename = "data/"+filename
    file1 = open(filename, "r")
    data = csv.reader(file1)
    next(data)
    seq_list = []
    label_list = []
    length_list = []
    for item in data:
        idx_seq = []
        seq = preprocess(item[1])
        label = item[0] 
        seq = seq.split()
        for word in seq:
            if word in word2idx.keys():
                idx_seq.append(word2idx[word])
            else:
                idx_seq.append(word2idx["unk"])
        length = len(idx_seq)
        length_list.append(length)
        seq_list.append(idx_seq)
        label_list.append(label)
    return seq_list, label_list, length_list

word2idx = torch.load("word2idx.pt")
idx2word = torch.load("idx2word.pt")
train_seqs, train_labels, train_lengths = make_dataset("train.csv", word2idx, idx2word)
test_seqs, test_labels, test_lengths = make_dataset("test.csv", word2idx, idx2word)

from torch.nn.utils.rnn import pad_sequence

train_seqs = pad_sequence([torch.LongTensor(i) for i in train_seqs], batch_first=True)
test_seqs = pad_sequence([torch.LongTensor(i) for i in test_seqs], batch_first=True)


train_labels = torch.LongTensor([int(i) for i in train_labels])
test_labels = torch.LongTensor([int(i) for i in test_labels])
train_lengths = torch.LongTensor(train_lengths)
test_lengths = torch.LongTensor(test_lengths)


print("train, test input shapes: ", train_seqs.shape, test_seqs.shape)
print("train, test label shapes: ", train_labels.shape, test_labels.shape)
print("train, test length shapes: ", train_lengths.shape, test_lengths.shape)

class classifier(nn.Module):
    def __init__(self, num_classes, in_dim):
        super(classifier, self).__init__()
        self.linear_layer = nn.Linear(in_dim, 600)
        self.lstm = nn.LSTM(600, 300, 1, batch_first=True)
        self.linear_layer2 = nn.Linear(300, num_classes)
    def forward(self, x, length):
        x = self.linear_layer(x)
        x, _ = self.lstm(x)
        # get the length'th hidden state from x
        #print(x.shape, "x shape")
        final_preds = []
        loc_cnt = 0
        for loc_len in length:
            loc_x = x[loc_cnt, loc_len-1, :]
            loc_out = self.linear_layer2(loc_x)
            loc_cnt += 1
            final_preds.append(loc_out)
        final_preds = torch.stack(final_preds)
        return final_preds
forward_lstm = torch.load("forward_model.pt").to(device)
backward_lstm = torch.load("backward_model.pt").to(device)

train_dataset = torch.utils.data.TensorDataset(train_seqs, train_labels, train_lengths)

# split train set into train and validation set 20 percent
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

test_dataset = torch.utils.data.TensorDataset(test_seqs, test_labels, test_lengths)

#train_dataset = train_dataset.to(device)
#test_dataset = test_dataset.to(device)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=True)
in_num = 1800
num_classes = 5
downstream_model = classifier(num_classes, in_num).to(device)
loss_func = nn.CrossEntropyLoss()

n_epochs = 15

pred_list = []
truth_list = []
downstream_model = torch.load("downstream_model.pt")
optimizer = optim.Adam(downstream_model.parameters(), lr=0.0001)
train_accurs = []
train_f1s = []
val_accurs = []
val_f1s = []
for epoch in range(n_epochs):
    train_accurs = []
    train_f1s = []
    val_accurs = []
    val_f1s = []
    print("Entering Training loop...")
    downstream_model.train()
    b1 = 0
    for data, label, length in train_loader:
        data = data.to(device)
        data_reverse = data.flip(1)
        _, (xf0, xf1, xf2) = forward_lstm(data)
        _, (xb0, xb1, xb2) = backward_lstm(data_reverse)
        # concat xf0 and xb0
        x0 = torch.cat((xf0, xb0), dim=2)
        x1 = torch.cat((xf1, xb1), dim=2)
        x2 = torch.cat((xf2, xb2), dim=2)
        #print(x0.shape, x1.shape, x2.shape)
        #input features is concat of x0, x1, x2
        input_features = torch.cat((x0, x1, x2), dim=2)
        output = downstream_model(input_features, length)
        # convert out to cpu
        output = output.to("cpu")
        label = label.to("cpu")
        loss = loss_func(output, label)
        correct_pred_class = torch.argmax(output, dim=1)
        pred_list.append(correct_pred_class)
        truth_list.append(label)
        # calculate accuracy, micro-f1 based on pred_list and truth_list
        accruacy = accuracy_score(label, correct_pred_class)
        micro_f1 = f1_score(label, correct_pred_class, average="micro")
        train_accurs.append(accruacy)
        train_f1s.append(micro_f1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # accuracy = sum of train_accurs / len(train_accurs)
        # micro_f1 = sum of train_f1s / len(train_f1s)
        accuracy = sum(train_accurs) / len(train_accurs)
        micro_f1 = sum(train_f1s) / len(train_f1s)
        print("Epoch: {}, Batch: {}, Train Accuracy: {}, Train Micro-F1: {}".format(epoch, b1, accruacy, micro_f1))
        b1+=1
    print("ENtering Eval loop...")
    downstream_model.eval()
    b2 = 0
    for data, label, length in val_loader:
        data = data.to(device)
        data_reverse = data.flip(1)
        _, (xf0, xf1, xf2) = forward_lstm(data)
        _, (xb0, xb1, xb2) = backward_lstm(data_reverse)
        # concat xf0 and xb0
        x0 = torch.cat((xf0, xb0), dim=2)
        x1 = torch.cat((xf1, xb1), dim=2)
        x2 = torch.cat((xf2, xb2), dim=2)
        #print(x0.shape, x1.shape, x2.shape)
        #input features is concat of x0, x1, x2
        input_features = torch.cat((x0, x1, x2), dim=2)
        output = downstream_model(input_features, length)
        # convert out to cpu
        output = output.to("cpu")
        label = label.to("cpu")
        loss = loss_func(output, label)
        correct_pred_class = torch.argmax(output, dim=1)
        pred_list.append(correct_pred_class)
        truth_list.append(label)
        # calculate accuracy, micro-f1 based on pred_list and truth_list
        accruacy = accuracy_score(label, correct_pred_class)
        micro_f1 = f1_score(label, correct_pred_class, average="micro")
        print("Epoch: ", )
        val_accurs.append(accruacy)
        val_f1s.append(micro_f1)
        accuracy = sum(train_accurs) / len(train_accurs)
        micro_f1 = sum(train_f1s) / len(train_f1s)
        print("Epoch: {}, Batch: {}, Val Accuracy: {}, Val Micro-F1: {}".format(epoch, b2, accruacy, micro_f1))
        b2+=1
    torch.save(downstream_model, "downstream_model.pt")

print("Entering Test loop...")
test_pred_list = []
test_truth_list = []
downstream_model.eval()
b3 = 0
test_accs = []
test_f1s = []
for data, label, length in test_loader:
    data = data.to(device)
    data_reverse = data.flip(1)
    _, (xf0, xf1, xf2) = forward_lstm(data)
    _, (xb0, xb1, xb2) = backward_lstm(data_reverse)
    # concat xf0 and xb0
    x0 = torch.cat((xf0, xb0), dim=2)
    x1 = torch.cat((xf1, xb1), dim=2)
    x2 = torch.cat((xf2, xb2), dim=2)
    #print(x0.shape, x1.shape, x2.shape)
    #input features is concat of x0, x1, x2
    input_features = torch.cat((x0, x1, x2), dim=2)
    output = downstream_model(input_features, length)
    # convert out to cpu
    output = output.to("cpu")
    label = label.to("cpu")
    loss = loss_func(output, label)
    correct_pred_class = torch.argmax(output, dim=1)
    test_pred_list.append(correct_pred_class)
    test_truth_list.append(label)
    # calculate accuracy, micro-f1 based on pred_list and truth_list
    accruacy = accuracy_score(label, correct_pred_class)
    micro_f1 = f1_score(label, correct_pred_class, average="micro")
    test_accs.append(accruacy)
    test_f1s.append(micro_f1)
    print("Epoch: {}, Batch: {}, Test Accuracy: {}, Test Micro-F1: {}".format(epoch, b3, accruacy, micro_f1))
    b3+=1

print("Train accuracy: ", sum(train_accurs)/len(train_accurs))
print("Train micro-f1: ", sum(train_f1s)/len(train_f1s))
print("Val accuracy: ", sum(val_accurs)/len(val_accurs))
print("Val micro-f1: ", sum(val_f1s)/len(val_f1s))
print("Test accuracy: ", sum(test_accs)/len(test_accs))
print("Test micro-f1: ", sum(test_f1s)/len(test_f1s))
torch.save(pred_list, "pred_list.pt")
torch.save(truth_list, "truth_list.pt")
torch.save(test_pred_list, "test_pred_list.pt")
torch.save(test_truth_list, "test_truth_list.pt")
        





