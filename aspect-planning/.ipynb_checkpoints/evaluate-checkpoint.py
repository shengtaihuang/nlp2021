import torch
from torch.autograd import Variable
import random
from model import *
from util import *
import sys
import os
from masked_cross_entropy import *
import itertools
import random
import math
from tqdm import tqdm
from load import SOS_ID, EOS_ID, PAD_ID
from model import AspectAttnDecoder, ContextEncoder
import pickle
import logging
from queue import PriorityQueue
import operator

logging.basicConfig(level=logging.INFO)
USE_CUDA = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hypothesis(object):
    def __init__(self, tokens, log_probs, state):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state

    def extend(self, token, log_prob, state):
        return Hypothesis(tokens = self.tokens + [token],
                        log_probs = self.log_probs + [log_prob],
                        state = state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property 
    def log_prob(self):
        return sum(self.log_probs)

    @property 
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)
 
def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.log_prob, reverse=True)

class BeamSearchNode(object):
    def __init__(self, hidden, prev_node, aspect_id, lop_p, length):
        self.hidden = hidden
        self.prev_node = prev_node
        self.aspect_id = aspect_id
        self.log_p = log_p
        self.len = length
        
        # Score should be normalized so that it serves as prob
        self.coherence = 0  # inject the term here
        self.generality = 0
        self.simplicity = 0

    def eval(self, beta, w):
        alpha = 1 - beta
        
        seq_log_p = self.log_p / float(self.len - 1 + 1e-6)
        additional_log_p = (w['coherence'] * self.coherence + \
                       w['generality'] * self.generality # coherence, generality are already log prob from PMI

        return (alpha * seq_log_p) + (beta * additional_log_p)

def conditional_beam_decode(decoder, decoder_hidden, beam_size, max_length, encoder_output=None):
#def beam_decode(decoder, target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor - batch information [B, T] (T: maximum output aspect length)
    :param decoder_hidden - input [1, B, H] as a starting point
    :param encoder_outputs
    :return decoded_batch
    '''

    beam_size = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []
    
    # parameter for additional factors
    beta = 0.4 # weight for all additional factors combined
    w = {
            'coherence': 0.2,
            'generality': 0.4,
            'simplicity': 0.4
        }

    # # decoding goes sentence by sentence
    # for idx in range(target_tensor.size(0)):
    # if isinstance(decoder_hiddens, tuple):  # LSTM case
    #     decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
    # else:
    #     decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
    #encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

    # Start with the start of the sentence token
    decoder_input = torch.LongTensor([[SOS_token]], device=device)

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, log_p, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(beta, w), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > 2000: break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.aspect_id
        decoder_hidden = n.h

        if n.aspect_id.item() == EOS_token and n.prev_node != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
        print('intput and output dim: ', decoder_input.shape, decoder_output.shape)

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(decoder_output, beam_size)
        print('indexes: ', indexes)
        print('log_prob dim: ', log_prob.shape)
        next_nodes = []

        for new_k in range(beam_size):
            decoded_t = indexes[0][0][new_k].view(1, -1)
            log_p = log_prob[0][0][new_k].item() # was "log_prob[0][new_k].item()"

            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.log_p + log_p, n.len + 1, beta)
            score = -node.eval(beta, w)
            next_nodes.append((score, node))

        # put them into queue
        for i in range(len(next_nodes)):
            score, nn = next_nodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(next_nodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.aspect_id)
        # back trace
        while n.prev_node != None:
            n = n.prev_node
            utterance.append(n.aspect_id)

        utterance = utterance[::-1]
        utterances.append(utterance)

    decoded_batch.append(utterances)

    return decoded_batch

def beam_decode(topic_decoder, decoder_hidden, encoder_out, vocab, beam_size, max_length, min_length):
    hyps = [Hypothesis(tokens=[SOS_ID], log_probs=[0.0], state=decoder_hidden) for _ in range(beam_size)] 
    results = [] 
    steps = 0
    
    print('hyps: ', hyps)
    print('max_length and beam size: ', max_length, beam_size)
    print('vocab.idx2topic: ', vocab.idx2topic)
    
    # go over timesteps one by one
    while steps < max_length and len(results) < beam_size:
        print('steps: ', steps)
        new_hiddens = []
        topk_ids = []
        topk_probs = []
        
        # hyp: candidate output sequence
        for hyp in hyps:
            print('hyp.tokens: ', hyp.tokens)
            decoder_input = Variable(torch.LongTensor([[hyp.latest_token]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
            
            decoder_hidden = hyp.state 
            
            decoder_output, decoder_hidden, _ = topic_decoder(decoder_input, decoder_hidden, encoder_out)
            print('intput and output dim: ', decoder_input.shape, decoder_output.shape)
            new_hiddens.append(decoder_hidden)
            
            # extract topk tokens
            topv, topi = decoder_output.data.topk(beam_size*2)
            topv = topv.squeeze(0)
            topi = topi.squeeze(0)
            print('topv: ', topv, topi)

            topk_ids.extend(topi)
            topk_probs.extend(torch.log(topv))

        all_hyps = []   
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        print('num_orig_hyps: ', num_orig_hyps)
        for i in range(num_orig_hyps):
            h, new_hidden = hyps[i], new_hiddens[i] 
            for j in range(beam_size*2):  
                new_hyp = h.extend(token=topk_ids[i][j], log_prob=topk_probs[i][j], state=new_hidden)
                all_hyps.append(new_hyp)

        hyps = []
        for h in sort_hyps(all_hyps): 
            if h.latest_token == EOS_ID: 
                if steps >= min_length:
                    results.append(h)
            else:
                hyps.append(h)
            if len(hyps) == beam_size or len(results) == beam_size:
                break

        steps += 1
        
    if len(results)==0: 
        results = hyps

    hyps_sorted = sort_hyps(results)
    print('hyps_sorted: ', hyps_sorted[0].tokens, hyps_sorted[0].log_probs)

    return hyps_sorted[0]


def decode(topic_decoder, decoder_hidden, encoder_out, vocab, max_length, min_length):

    decoder_input = Variable(torch.LongTensor([[SOS_ID]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden, _ = topic_decoder(decoder_input, decoder_hidden, encoder_out)

        topv, topi = decoder_output.data.topk(4)
        topi = topi.squeeze(0)
        topv = topv.squeeze(0)
        ni = topi[0][0]
        if ni == EOS_ID:
            decoded_words.append('<eos>')
            break
        else:
            decoded_words.append(vocab.idx2topic[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    if decoded_words != '<eos>':
        decoded_words.append('<eos>')

    return decoded_words


def evaluate(encoder, topic_decoder, vocab, pair, beam_size, max_length, min_length):
    sentence = pair # (user_id, item_id, rating)
    attr_input = Variable(torch.LongTensor([sentence]), volatile=True)
    attr_input = attr_input.cuda() if USE_CUDA else attr_input

    encoder_out, encoder_hidden = encoder(attr_input) 
    decoder_hidden = encoder_hidden[:topic_decoder.n_layers]
    
    if beam_size == 1:
        return decode(topic_decoder, decoder_hidden, encoder_out, vocab, max_length, min_length)
    else:
        #return beam_decode(topic_decoder, decoder_hidden, encoder_out, vocab, beam_size, max_length, min_length)
        return conditional_beam_decode(topic_decoder, decoder_hidden, beam_size, max_length, encoder_out)


def evaluateRandomly(encoder, topic_decoder, vocab, user_rdict, item_rdict, pairs, n_pairs, beam_size, max_length, min_length, save_dir):

    path = os.path.join(save_dir, 'decode')
    if not os.path.exists(path):
        os.makedirs(path)
    f1 = open(path + "/decoded.txt", 'w')
    for i in range(n_pairs):
    
        pair = pairs[i]
        
        user = pair[0][0]
        item = pair[0][1]
        rating = pair[0][2]
        topic = ' '.join(pair[1][1:-1])
        print("=============================================================")
        print("user_rdict: ", user_rdict.keys())
        print("item_rdict: ", item_rdict.keys())
        print('Attribute > ', '\t'.join([user_rdict[user], item_rdict[item], str(rating+1)]))
        print('Reference > ', topic)

        f1.write('Context: ' + '\t'.join([user_rdict[user], item_rdict[item], str(rating+1)]) + '\n' + 'Reference: ' + topic + '\n')
        if beam_size == 1:
            output_words = evaluate(encoder, topic_decoder, vocab, pair[0], beam_size, max_length, min_length)
            output_sentence = ' '.join(output_words[:-1])
            print('<', output_sentence)
            f1.write('Generation: ' + output_sentence + "\n")
        else:
            # best_hyp.tokens originally looks like :  [0, tensor(15), tensor(75), tensor(75), tensor(75), tensor(75), tensor(125), tensor(125), tensor(170), tensor(170), tensor(75)]
            best_hyp = evaluate(encoder, topic_decoder, vocab, pair[0], beam_size, max_length, min_length)
            output_idx = [int(t) for t in best_hyp.tokens]
            print('idx2topic: ', vocab.idx2topic)
            print('output_idx: ', output_idx)
            output_words = [vocab.idx2topic[idx] for idx in output_idx]
            if output_words[-1] != '<eos>':
                output_words.append('<eos>')
            output_sentence = ' '.join(output_words[1:-1])
            f1.write('Generation: ' + output_sentence + '\n')
            print("Generation > ", output_sentence)
    f1.close()

def runTest(corpus, n_layers, hidden_size, embed_size, attr_size, attr_num, overall, modelFile, beam_size, max_length, min_length, save_dir):

    vocab, train_pairs, valid_pairs, test_pairs = loadPrepareData(corpus, save_dir)
    
    print('Building encoder and decoder ...')
    with open(os.path.join(save_dir, 'user_rev.pkl'), 'rb') as fp:
        user_rdict = pickle.load(fp)
    with open(os.path.join(save_dir, 'item_rev.pkl'), 'rb') as fp:
        item_rdict = pickle.load(fp)
        
    num_users = len(user_rdict)
    num_items = len(item_rdict)
    max_rating = overall 
    
    context_embeddings = []
    context_embeddings.append(nn.Embedding(num_users, attr_size))
    context_embeddings.append(nn.Embedding(num_items, attr_size))
    context_embeddings.append(nn.Embedding(max_rating, attr_size))
    
    if USE_CUDA:
        for attr_embedding in context_embeddings:
            attr_embedding = attr_embedding.cuda()

    encoder = ContextEncoder(attr_size, attr_num, hidden_size, context_embeddings, n_layers)

    topic_embedding = nn.Embedding(vocab.n_topics, embed_size)   

    if USE_CUDA:
        topic_embedding = topic_embedding.cuda()

    topic_decoder = AspectAttnDecoder(topic_embedding, embed_size, hidden_size, attr_size, vocab.n_topics, n_layers)
    
    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['encoder'])
    topic_decoder.load_state_dict(checkpoint['topic_decoder'])

    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        topic_decoder = topic_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False)
    topic_decoder.train(False)

    evaluateRandomly(encoder, topic_decoder, vocab, user_rdict, item_rdict, test_pairs, len(test_pairs), beam_size, max_length, min_length, save_dir)