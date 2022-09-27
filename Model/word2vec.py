import os, sys
import numpy as np
import EMdata

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from numpy.random import multinomial
import itertools

import random, math
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import time

print('all modules imported')


# all the functions

def cut_corpus(corpus,cut_length):
    new_corpus=[]
    cut_length=cut_length
    print(len(corpus))
    for i in range(len(corpus)):
        lst=corpus[i]
        n=len(lst)
        if n<cut_length:
            new_corpus.append(lst)
            continue
        cut_amount=int((n-n%cut_length)/cut_length)
        for j in range(cut_amount-1):
            new_corpus.append(lst[j*cut_length:(j+1)*cut_length])
        new_corpus.append(lst[(cut_amount-1)*cut_length:])
    print(len(new_corpus))
    return new_corpus


# word2vec function
def sample_negative(sample_size):
    sample_probability = {}
    word_counts = dict(Counter(list(itertools.chain.from_iterable(corpus_ignore))))
    normalizing = sum([v**0.75 for v in word_counts.values()])
    for word in word_counts:
        sample_probability[word] = word_counts[word]**0.75 / normalizing
    words = np.array(list(word_counts.keys()))
    while True:
        word_list = []
        sampled_index = np.array(multinomial(sample_size, list(sample_probability.values())))
        for index, count in enumerate(sampled_index):
            for _ in range(count):
                 word_list.append(words[index])
        yield word_list

def get_batches(context_tuple_list, batch_size=100):
    random.shuffle(context_tuple_list)
    batches = []
    batch_target, batch_context, batch_negative = [], [], []
    for i in range(len(context_tuple_list)):
        batch_target.append(word_to_index[context_tuple_list[i][0]])
        batch_context.append(word_to_index[context_tuple_list[i][1]])
        batch_negative.append([word_to_index[w] for w in context_tuple_list[i][2]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = torch.from_numpy(np.array(batch_target)).long().to(device)
            tensor_context = torch.from_numpy(np.array(batch_context)).long().to(device)
            tensor_negative = torch.from_numpy(np.array(batch_negative)).long().to(device)
            batches.append((tensor_target, tensor_context, tensor_negative))
            batch_target, batch_context, batch_negative = [], [], []
    return batches

class Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.target = nn.Embedding(vocab_size, embedding_size).cuda()
        self.context = nn.Embedding(vocab_size, embedding_size).cuda()

    def forward(self, target_word, context_word, negative_example):
        emb_target = self.target(target_word)
        emb_context = self.context(context_word)
        emb_product = torch.mul(emb_target, emb_context).cuda()
        emb_product = torch.sum(emb_product, dim=1).cuda()
        out = torch.sum(F.logsigmoid(emb_product)).cuda()
        emb_negative = self.context(negative_example)
        emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2)).cuda()
        emb_product = torch.sum(emb_product, dim=1).cuda()
        out += torch.sum(F.logsigmoid(-emb_product)).cuda()
        return -out

class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_gain = min_percent_gain / 100.
        
    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]
    
    def stop(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)
        print("Loss gain: {}%".format(gain*100))
        if gain < self.min_gain:
            return True
        else:
            return False


#running model
def run_model(corpus_ignore , embedding_size=100, w = 2):
    #create vocabulary and its index
    vocabulary = set(itertools.chain.from_iterable(corpus_ignore))
    vocabulary_size = len(vocabulary)
    print(vocabulary_size)

    word_to_index = {w: idx for (idx, w) in enumerate(vocabulary)}
    index_to_word = {idx: w for (idx, w) in enumerate(vocabulary)}

    # create to 2D class pairs
    context_tuple_list = []
    negative_samples = sample_negative(4)

    for text in corpus_ignore:
        for i, word in enumerate(text):
            if word==0:
                print(type(word))
                continue
            first_context_word_index = max(0,i-w)
            last_context_word_index = min(i+w, len(text))
            for j in range(first_context_word_index, last_context_word_index):
                neighbor=text[j]
                if j==0:
                    continue
                if neighbor==0:
                    continue
                if i!=j:
                    context_tuple_list.append((word, neighbor, next(negative_samples)))
    print("There are {} pairs of target and context words".format(len(context_tuple_list)))


    net = Word2Vec(embedding_size=embedding_size, vocab_size=vocabulary_size)
    optimizer = optim.Adam(net.parameters())
    early_stopping = EarlyStopping(patience=5, min_percent_gain=1)
    n=0
    while True:
        n=n+1
        print(n)
        losses = []
        context_tuple_batches = get_batches(context_tuple_list, batch_size=1000)
        for i in range(len(context_tuple_batches)):
            net.zero_grad()
            target_tensor, context_tensor, negative_tensor = context_tuple_batches[i]
            loss = net(target_tensor, context_tensor, negative_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
        print("Loss: ", torch.mean(torch.stack(losses)),torch.stack(losses)[0],torch.stack(losses)[-1])
        early_stopping.update_loss(torch.mean(torch.stack(losses)))
        if early_stopping.stop() is True:
            break
        if torch.mean(torch.stack(losses))<10:
            break

    #2D class embedding
    EMBEDDINGS = net.target.weight.data.cpu().numpy()
    EMBEDDINGS_np=np.array(EMBEDDINGS)

    # average filament embedding

    average_method=0 # 0 is average, 1 is weight average
    filament_score=[]
    all_filament_data=[]
    filament_variance=[]
    for filament in corpus_ignore:
        score=torch.zeros(embedding_size)
        counts=0
        filament_list=[]
        for i in filament:
            if i==0:
                continue
            counts+=1
            filament_list.append(EMBEDDINGS[word_to_index[i]])
        if len(filament_list)==0:
            print('no')
        filament_list=np.array(filament_list)
        if len(filament_list)==1:
            filament_variance.append(float(0))
        else:
            pca=PCA(n_components=1).fit(filament_list)
            filament_variance.append(pca.singular_values_[0])
        mean=filament_list.mean(axis=0)
        all_filament_data.append(filament_list)
        if average_method==0:
            filament_score.append(mean)
        elif average_method==1:
            dim=len(filament_list[0])
            filament_normalized=np.exp(-0.5*((filament_list-mean) @ (filament_list-mean).T*0)).diagonal()/np.sqrt(np.pi**dim*0.05)
            filament_normalized=filament_normalized/filament_normalized.sum()
            score=filament_normalized @ filament_list
            if counts<=2:
                continue
            filament_score.append(np.array(score))
    all_data=filament_score[:]
    all_data.extend(EMBEDDINGS_np)
    filament_number=len(filament_score)
    print('The filament number are: ',filament_number)
    return all_data