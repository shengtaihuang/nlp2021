import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm
from queue import PriorityQueue
import sys, os, random, itertools, math, operator, pickle, logging

from model import *
from util import *
from masked_cross_entropy import *
from load import SOS_ID, EOS_ID, PAD_ID
from model import AspectAttnDecoder, ContextEncoder


SOS_token = 0
EOS_token = 1
#USE_CUDA = torch.cuda.is_available()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#logging.basicConfig(level=logging.INFO)

USE_CUDA = 0
device = "cpu"

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
    def __init__(self, hidden, prev_node, aspect_id, log_p, length):
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
        additional_log_p = w['coherence'] * self.coherence + \
                       w['generality'] * self.generality # coherence, generality are already log prob-ed from PMI

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
    topk = 5  # how many sentence do you want to generate
    decoded_batch = []
    
    # parameter for additional factors
    beta = 0.2 # weight for all additional factors combined
    w = {
            'coherence': 0.2,
            'generality': 0.4,
            'simplicity': 0.4
        }
    print('Evaluate with conditional_beam_decoder with beta={}, with weights={}'.format(beta, w))
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
        decoder_hidden = n.hidden

        if n.aspect_id.item() == EOS_token and n.prev_node != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)

        log_prob, indexes = torch.topk(decoder_output, beam_size)
        next_nodes = []

        # Iterate over candidate tokens
        for new_k in range(beam_size):
            decoded_t = indexes[0][0][new_k].view(1, -1)
            log_p = log_prob[0][0][new_k].item() # was "log_prob[0][new_k].item()"

            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.log_p + log_p, n.len + 1)
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

    decoded = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        hyp = []
        hyp.append(n.aspect_id)

        while n.prev_node != None:
            n = n.prev_node
            hyp.append(n.aspect_id)
        hyp = hyp[::-1]
        decoded.append(hyp)
        
    return decoded[0]

def beam_decode(aspect_decoder, decoder_hidden, encoder_out, vocab, beam_size, max_length, min_length):
    hyps = [Hypothesis(tokens=[SOS_ID], log_probs=[0.0], state=decoder_hidden) for _ in range(beam_size)] 
    results = [] 
    steps = 0
    beam_topk = []
    
    #print('hyps: ', hyps.tokens)
    print('max_length and beam size: ', max_length, beam_size)
    
    # go over timesteps one by one
    while steps < max_length and len(results) < beam_size:
        new_hiddens = []
        topk_ids = []
        topk_probs = []
        
        beam_topk_aspects = []
        beam_topk_probs = []
        # hyp: candidate output sequence
        for hyp in hyps:
            decoder_input = Variable(torch.LongTensor([[hyp.latest_token]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input
            
            decoder_hidden = hyp.state 
            
            decoder_output, decoder_hidden, _ = aspect_decoder(decoder_input, decoder_hidden, encoder_out)
            #print('intput and output dim: ', decoder_input.shape, decoder_output.shape)
            new_hiddens.append(decoder_hidden)
            
            # extract topk tokens
            topv, topi = decoder_output.data.topk(beam_size*2)
            topv = topv.squeeze(0)
            topi = topi.squeeze(0)
            

            topk_ids.extend(topi)
            topk_probs.extend(torch.log(topv))
            #print('topv: ', topi, topv)
            beam_topk_aspects.extend([ vocab.idx2aspect[idx] for idx in list(topi[0].numpy()) ])
            #print('topk_probs: ', topk_probs)
            beam_topk_probs.extend([ float(p) for p in list(topv[0].numpy()) ])
        
        all_hyps = []   
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        #print('num_orig_hyps: ', num_orig_hyps)
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
        
    beam_topk = {'word': beam_topk_aspects, 'logp': beam_topk_probs}
    if len(results)==0: 
        results = hyps

    hyps_sorted = sort_hyps(results)
    
    #for h in hyps_sorted:
    #    print('hyps_sorted: ', [vocab.idx2aspect[int(t)] for t in h.tokens], h.log_probs)

    
    return hyps_sorted[0].tokens, hyps_sorted[0].log_probs, beam_topk


def decode(aspect_decoder, decoder_hidden, encoder_out, vocab, max_length, min_length):

    decoder_input = Variable(torch.LongTensor([[SOS_ID]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []
    for di in range(max_length):
        decoder_output, decoder_hidden, _ = aspect_decoder(decoder_input, decoder_hidden, encoder_out)

        topv, topi = decoder_output.data.topk(4)
        topi = topi.squeeze(0)
        topv = topv.squeeze(0)
        ni = topi[0][0]
        print('ni: ', ni)
        if ni == EOS_ID:
            #decoded_words.append('<eos>')
            decoded_words.append(ni)
            break
        else:
            #decoded_words.append(vocab.idx2aspect[ni])
            decoded_words.append(ni)

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    #if decoded_words != '<eos>':
    #    decoded_words.append('<eos>')
    if decoded_words != EOS_ID:
        decoded_words.append(EOS_ID)

    return decoded_words


def evaluate(encoder, aspect_decoder, vocab, pair, beam_size, max_length, min_length):
    sentence = pair # (user_id, item_id, rating)
    attr_input = Variable(torch.LongTensor([sentence]), volatile=True)
    attr_input = attr_input.cuda() if USE_CUDA else attr_input

    encoder_out, encoder_hidden = encoder(attr_input) 
    decoder_hidden = encoder_hidden[:aspect_decoder.n_layers]
    
    if beam_size == 1:
        print('Evaluate with decoder')
        return decode(aspect_decoder, decoder_hidden, encoder_out, vocab, max_length, min_length)
    else:
        #print('Evaluate with beam_decoder')
        #return beam_decode(aspect_decoder, decoder_hidden, encoder_out, vocab, beam_size, max_length, min_length)
        
        return conditional_beam_decode(aspect_decoder, decoder_hidden, beam_size, max_length, encoder_out)


def evaluateRandomly(encoder, aspect_decoder, vocab, user_rdict, item_rdict, pairs, n_pairs, beam_size, max_length, min_length, save_dir):

    path = os.path.join(save_dir, 'decode')
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + "/decoded.txt", 'w')
    
    output_words_logp = []
    beam_output_words_logp = []
    for i in range(n_pairs)[:10]:
    
        pair = pairs[i]
        
        context_input = pair[0] # user,item,rating
        user = pair[0][0]
        item = pair[0][1]
        rating = pair[0][2]
        aspect = ' '.join(pair[1][1:-1])
        print("=============================================================")
        #print("user_rdict: ", user_rdict.keys())
        #print("item_rdict: ", item_rdict.keys())
        print('Context: ', '\t'.join([user_rdict[user], item_rdict[item], str(rating+1)]))
        print('Aspects > ', aspect)

        f.write('Context: ' + '\t'.join([user_rdict[user], item_rdict[item], str(rating+1)]) + '\n' + 'Ground-truth aspect: ' + aspect + '\n')

        best_hyp = evaluate(encoder, aspect_decoder, vocab, context_input, beam_size, max_length, min_length)
        
        # For experimental results
#         output_logp = [float(t) for t in best_hyp_logp] 

        output_idx = [int(t) for t in best_hyp]
        output_words = [vocab.idx2aspect[idx] for idx in output_idx]
#         output_words_logp.append([output_words, output_logp])
#         beam_output_words_logp.append(beam_topk)
        
        if output_words[-1] != '<eos>':
            output_words.append('<eos>')
        output_sentence = ' '.join(output_words[1:-1])
        f.write('Predicted aspects: ' + output_sentence + '\n')
        print("Predicted aspects: ", output_sentence)
    
#     df_beam = pd.DataFrame(beam_output_words_logp, columns=['word', 'logp'])
#     df_output = pd.DataFrame(output_words_logp, columns=['word', 'logp'])
#     df_beam.to_csv(path + "/beam_output_words_logp.csv", index=False)
#     df_output.to_csv(path + "/output_words_logp.csv", index=False)
    f.close()

def runTest(corpus, n_layers, hidden_size, embed_size, context_embed_size, num_contexts, overall, modelFile, beam_size, max_length, min_length, save_dir):
    beam_size = 10
    vocab, train_pairs, valid_pairs, test_pairs = loadPrepareData(corpus, save_dir)
    
    #torch.cuda.set_device(6)
    print('testing the model "{}"'.format(modelFile))
    print('Building encoder and decoder ...')
    print('cuda is available? {}(device id: {}) (among {} available ones)'.format(USE_CUDA, torch.cuda.current_device(), torch.cuda.device_count()))
    
    with open(os.path.join(save_dir, 'idx2user.pkl'), 'rb') as fp:
        user_rdict = pickle.load(fp)
    with open(os.path.join(save_dir, 'idx2item.pkl'), 'rb') as fp:
        item_rdict = pickle.load(fp)
        
    num_users = len(user_rdict)
    num_items = len(item_rdict)
    max_rating = overall 
    
    context_embeddings = []
    context_embeddings.append(nn.Embedding(num_users, context_embed_size))
    context_embeddings.append(nn.Embedding(num_items, context_embed_size))
    context_embeddings.append(nn.Embedding(max_rating, context_embed_size))
    
    if USE_CUDA:
        for attr_embedding in context_embeddings:
            attr_embedding = attr_embedding.cuda()

    encoder = ContextEncoder(context_embed_size, num_contexts, hidden_size, context_embeddings, n_layers)

    aspect_emb = nn.Embedding(vocab.n_topics, embed_size)   

    if USE_CUDA:
        aspect_emb = aspect_emb.cuda()

    aspect_decoder = AspectAttnDecoder(aspect_emb, embed_size, hidden_size, context_embed_size, vocab.n_topics, n_layers)
    
    print('encoder: ', encoder.attr.weight.device)
    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['encoder'])
    aspect_decoder.load_state_dict(checkpoint['aspect_decoder'])

    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        aspect_decoder = aspect_decoder.cuda()

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False)
    aspect_decoder.train(False)

    evaluateRandomly(encoder, aspect_decoder, vocab, user_rdict, item_rdict, test_pairs, len(test_pairs), beam_size, max_length, min_length, save_dir)
    #print('encoder: ', encoder.attr.weight.device)