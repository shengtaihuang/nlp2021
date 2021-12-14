import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
from masked_cross_entropy import *

import numpy as np
import itertools
import random
import math
import sys
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_ID, EOS_ID, PAD_ID
import pickle
import logging
logging.basicConfig(level=logging.INFO)
    
    
def indexesFromSentence(vocab, sentence): # for output as aspect
    ids = []
    for word in sentence:
        word = word
        ids.append(vocab.aspect2idx[word])
    return ids

def Padding(l, fillvalue=PAD_ID):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue)) 

def binaryMatrix(l, value=PAD_ID):
    m = []
    for i in range(len(l)):
        m.append([])
        for j in range(len(l[i])):
            if l[i][j] == PAD_ID:
                m[i].append(0)
            else:
                m[i].append(1) # mask = 1 if not padding
    return m

# return context index 
def context_variable(data, vocab, evaluation=False):
    
    context = [[d[0], d[1], d[2]] for d in data]  # length: batch, # d[0]: list, d[1]: list, d[2]: 4.0     
    context_variable = Variable(torch.LongTensor(context), volatile=evaluation) # (batch_size, attribute_num)

    return context_variable 

# convert to index, add zero padding
# return output variable, mask, max length of the sentences in batch
def output_variable(l, vocab):
    aspect_input = [indexesFromSentence(vocab, sentence[:-1]) for sentence in l] # Go over each output in a batch
    aspect_output = [indexesFromSentence(vocab, sentence[1:]) for sentence in l]
    inpadList = Padding(aspect_input)
    outpadList = Padding(aspect_output)
    mask = binaryMatrix(inpadList)
    mask = Variable(torch.ByteTensor(mask))
    inpadVar = Variable(torch.LongTensor(inpadList))
    outpadVar = Variable(torch.LongTensor(outpadList))
    return inpadVar, outpadVar, mask

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by output length, reverse input
def batch2TrainData(vocab, pair_batch, evaluation=False):
    pair_batch.sort(key=lambda x: len(x[1]), reverse=True) # sort on topic number
    input_batch, output_batch = [], []
    for i in range(len(pair_batch)):
        input_batch.append(pair_batch[i][0])
        output_batch.append(pair_batch[i][1])
        
    context_input = context_variable(input_batch, vocab, evaluation=evaluation)
    aspect_input, aspect_output, mask = output_variable(output_batch, vocab) # convert sentence to ids and padding
    return context_input, aspect_input, aspect_output, mask

