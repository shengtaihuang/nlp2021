# raise ValueError("deal with Variable requires_grad, and .cuda()")
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter

import numpy as np 
import itertools
import random
import math
import sys
import os
import pickle
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_ID, EOS_ID, PAD_ID
from model import ContextEncoder, AspectAttnDecoder
from util import batch2TrainData
import time
from masked_cross_entropy import *

cudnn.benchmark = True
USE_CUDA = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4,5,6,7'

#############################################
# Training
#############################################

def adjust_learning_rate(optimizer, epoch, learning_rate, lr_decay_epoch, lr_decay_ratio):
    lr = learning_rate * (lr_decay_ratio ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def from_pretrained(embeddings, freeze=True):
    assert embeddings.dim() == 2, 'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embeddings)
    embedding.weight.requires_grad = not freeze
    return embedding

def train(attr_input, aspect_input, aspect_output, mask, encoder, aspect_decoder, encoder_optimizer, aspect_decoder_optimizer):
    encoder_optimizer.zero_grad()   
    aspect_decoder_optimizer.zero_grad()    

    if USE_CUDA:
        attr_input = attr_input.cuda() 
        aspect_input = aspect_input.cuda()
        aspect_output = aspect_output.cuda()
        mask = mask.cuda()

    # title encoder
    encoder_out, encoder_hidden = encoder(attr_input) # attribute encoder

    decoder_input = aspect_input  
    decoder_hidden = encoder_hidden[:aspect_decoder.n_layers]
    
    # Run through decoder one time step at a time
    print_losses = []
    loss = 0.0

    decoder_output, decoder_hidden, decoder_attn = aspect_decoder(decoder_input, decoder_hidden, encoder_out)
    
    mask_loss = masked_cross_entropy(decoder_output, aspect_output, mask)
    loss += mask_loss
    #print_losses.append(mask_loss.data[0])
    print_losses.append(mask_loss.data)
    
    loss.backward()  # BP process

    clip = 5.0
    ec = torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, encoder.parameters()), clip)
    dc = torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, aspect_decoder.parameters()), clip)

    encoder_optimizer.step()
    aspect_decoder_optimizer.step()

    return sum(print_losses) 

def evaluate(attr_input, aspect_input, aspect_output, mask, encoder, aspect_decoder, encoder_optimizer, aspect_decoder_optimizer):

    encoder.eval()
    aspect_decoder.eval()

    if USE_CUDA:
        attr_input = attr_input.cuda() 
        aspect_input = aspect_input.cuda()
        aspect_output = aspect_output.cuda()
        mask = mask.cuda()

    encoder_out, encoder_hidden = encoder(attr_input) # attribute encoder

    decoder_input = aspect_input
    decoder_hidden = encoder_hidden[:aspect_decoder.n_layers]

    # Run through decoder one time step at a time
    print_losses = []
    loss = 0

    decoder_output, decoder_hidden, decoder_attn = aspect_decoder(decoder_input, decoder_hidden, encoder_out)
    mask_loss = masked_cross_entropy(decoder_output, aspect_output, mask)
    loss += mask_loss
    #print_losses.append(mask_loss.data[0])
    print_losses.append(mask_loss.data)

    return sum(print_losses) 

def batchify(pairs, bsz, vocab, evaluation=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(pairs) // bsz
    data = []
    for i in range(nbatch):
        batch = batch2TrainData(vocab, pairs[i * bsz: i * bsz + bsz], evaluation)
        data.append(batch)
    return data

def trainIters(corpus, learning_rate, lr_decay_epoch, lr_decay_ratio, batch_size, n_layers, hidden_size, embed_size, 
        context_emb_size, num_contexts, overall, save_dir, loadFilename=None):

    print("corpus={}, learning_rate={}, lr_decay_epoch={}, lr_decay_ratio={}, batch_size={}, n_layers={}, \
    hidden_size={}, embed_size={}, context_emb_size={}, num_contexts={}, overall={}, save_dir={}".format(corpus, learning_rate, lr_decay_epoch, \
    lr_decay_ratio, batch_size, n_layers, hidden_size, embed_size, context_emb_size, num_contexts, overall, save_dir))
    torch.cuda.set_device(7)
    print('cuda is available? {}(device id: {}) (among {} available ones)'.format(USE_CUDA, torch.cuda.current_device(), torch.cuda.device_count()))
    
    print('load data...')
    vocab, train_pairs, valid_pairs, test_pairs = loadPrepareData(corpus, save_dir)  # load data
    print('data loaded...')

    data_path = os.path.join(save_dir, "batches")
    
    # training data
    corpus_name = corpus
    training_batches = None
    try:
        training_batches = torch.load(os.path.join(data_path, '{}_{}.tar'.format('training_batches', batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = batchify(train_pairs, batch_size, vocab)
        print('Complete building training pairs ...')
        torch.save(training_batches, os.path.join(data_path, '{}_{}.tar'.format('training_batches', batch_size)))    

    # validation/test data
    eval_batch_size = 10
    try:
        val_batches = torch.load(os.path.join(data_path, '{}_{}.tar'.format('val_batches', eval_batch_size)))
    except FileNotFoundError:
        print('Validation pairs not found, generating ...')
        val_batches = batchify(valid_pairs, eval_batch_size, vocab, evaluation=True)  
        print('Complete building validation pairs ...')
        torch.save(val_batches, os.path.join(data_path, '{}_{}.tar'.format('val_batches', eval_batch_size)))

    try:
        test_batches = torch.load(os.path.join(data_path, '{}_{}.tar'.format('test_batches', eval_batch_size)))
    except FileNotFoundError:
        print('Test pairs not found, generating ...')
        test_batches = batchify(test_pairs, eval_batch_size, vocab, evaluation=True)
        print('Complete building test pairs ...')
        torch.save(test_batches, os.path.join(data_path, '{}_{}.tar'.format('test_batches', eval_batch_size)))

    # model
    checkpoint = None 
    print('Building encoder and decoder ...')
    
    # attribute embeddings
    with open(os.path.join(save_dir, 'user2idx.pkl'), 'rb') as fp:
        user_dict = pickle.load(fp)
    with open(os.path.join(save_dir, 'item2idx.pkl'), 'rb') as fp:
        item_dict = pickle.load(fp)
        
    num_users = len(user_dict)
    num_items = len(item_dict)
    max_rating = overall 
    
    context_embs = []
    u_embedding = nn.Embedding(num_users, context_emb_size)
    context_embs.append(u_embedding)

    i_embedding = nn.Embedding(num_items, context_emb_size)
    context_embs.append(i_embedding)
    
    cat_mat = torch.cat((torch.eye(max_rating), torch.zeros(max_rating, context_emb_size-max_rating)), dim=1)
    r_embedding = from_pretrained(torch.cat((torch.eye(max_rating), torch.zeros(max_rating, context_emb_size-max_rating)), dim=1)) # required_grad = False
    context_embs.append(r_embedding)
    
    if USE_CUDA:
        for context_emb in context_embs:
            context_emb = context_emb.cuda()

    encoder = ContextEncoder(context_emb_size, num_contexts, hidden_size, context_embs, n_layers) 

    # topic embedding
    t_emb = nn.Embedding(vocab.n_topics, embed_size)
   
    if USE_CUDA:
        t_emb = t_emb.cuda()

    attn_model = 'dot'
    aspect_decoder = AspectAttnDecoder(t_emb, embed_size, hidden_size, context_emb_size, vocab.n_topics, n_layers)
    
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['encoder'])
        aspect_decoder.load_state_dict(checkpoint['aspect_decoder'])
        
    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        aspect_decoder = aspect_decoder.cuda()
        
    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate)
    aspect_decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, aspect_decoder.parameters()), lr=learning_rate)  
    
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['encoder_opt'])
        aspect_decoder_optimizer.load_state_dict(checkpoint['aspect_decoder_opt'])

    # initialize
    print('Initializing ...')
    step = 0
    epoch = 0
    perplexity = []
    _loss = []
    
    log_path = os.path.join('ckpt/' + corpus_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    
    best_val_loss = None
    
    if loadFilename:
        step = checkpoint['step']
        epoch = checkpoint['epoch'] + 1
        perplexity = checkpoint['plt']
        _loss = checkpoint['loss']
        for i in range(len(_loss)):
            writer.add_scalar("Train/loss", _loss[i], i)
            writer.add_scalar("Train/perplexity", perplexity[i], i)

    while True:
            
        # learning rate adjust
        #adjust_learning_rate(encoder_optimizer, epoch, learning_rate, lr_decay_epoch, lr_decay_ratio)
        #adjust_learning_rate(aspect_decoder_optimizer, epoch, learning_rate, lr_decay_epoch, lr_decay_ratio)
        
        # train epoch
        encoder.train()
        aspect_decoder.train()
        
        tr_loss = 0   
        for batch_idx, training_batch in enumerate(training_batches):
            attr_input, aspect_input, aspect_output, mask = training_batch

            loss = train(attr_input, aspect_input, aspect_output, mask, encoder, aspect_decoder, encoder_optimizer, aspect_decoder_optimizer)
            step += 1
            
            tr_loss += loss

            _loss.append(loss)
            perplexity.append(math.exp(loss))    
            
            writer.add_scalar("Train/loss", loss, step)
            writer.add_scalar("Train/perplexity", math.exp(loss), step)
            
            print("epoch {} batch {} loss={} perplexity={} en_lr={:05.5f} de_lr={:05.5f}".format(epoch, batch_idx, loss, 
            math.exp(loss), encoder_optimizer.param_groups[0]['lr'], aspect_decoder_optimizer.param_groups[0]['lr']))
            break
            
        cur_loss = tr_loss / len(training_batches)
        
        print('\n' + '-' * 30)
        print('train | epoch {:3d} | average loss {:5.5f} | average ppl {:8.3f}'.format(epoch, cur_loss, math.exp(cur_loss)))
        print('-' * 30)

        print_loss = 0
            
        # evaluate
        vl_loss = 0
        for val_batch in val_batches:
            attr_input, aspect_input, aspect_output, mask = val_batch
            
            loss = evaluate(attr_input, aspect_input, aspect_output, mask, encoder, aspect_decoder, encoder_optimizer, aspect_decoder_optimizer)
            
            vl_loss += loss
        vl_loss /= len(val_batches)
        
        #return
        
        writer.add_scalar("Valid/loss", vl_loss, step)

        print('\n' + '-' * 30)
        print('valid | epoch {:3d} | valid loss {:5.5f} | valid ppl {:8.3f}'.format(epoch, vl_loss, math.exp(vl_loss)))
        print('-' * 30)
        
        model_path = os.path.join(save_dir, "model")
        if not best_val_loss or vl_loss < best_val_loss:
            directory = os.path.join(model_path, '{}_{}_{}'.format(n_layers, hidden_size, batch_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'step': step,
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'aspect_decoder': aspect_decoder.state_dict(),
                'encoder_opt': encoder_optimizer.state_dict(),
                'aspect_decoder_opt': aspect_decoder_optimizer.state_dict(),
                'loss': _loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'aspect_planning')))
            best_val_loss = vl_loss
     
            # Run on test data.
            ts_loss = 0
            for test_batch in test_batches:
                attr_input, aspect_input, aspect_output, mask = test_batch

                loss = evaluate(attr_input, aspect_input, aspect_output, mask, encoder, aspect_decoder, encoder_optimizer, aspect_decoder_optimizer)
                
                ts_loss += loss
            ts_loss /= len(test_batches)
            writer.add_scalar("Test/loss", ts_loss, step)
            
            print('\n' + '-' * 30)
            print('| test loss {:5.2f} | test ppl {:8.2f}'.format(ts_loss, math.exp(ts_loss)))
            print('-' * 30 + '\n')

        
        if vl_loss > best_val_loss:
            print('validation loss is larger than best validation loss. Break!')
            break

        epoch += 1

