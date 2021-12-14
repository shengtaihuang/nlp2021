# coding=utf-8
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
        
class ContextEncoder(nn.Module):
    def __init__(self, context_emb_size, num_contexts, hidden_size, context_embs, n_layers=1):
        super(ContextEncoder, self).__init__()
        self.context_emb_size = context_emb_size
        self.num_contexts = num_contexts  
        self.hidden_size = hidden_size
        self.n_layers = n_layers 
        self.u_embedding = context_embs[0]
        self.i_embedding = context_embs[1]
        self.r_embedding = context_embs[2]
        # hidden matrix is L*n, where L is number of layers and n is hidden size of each unit
        self.attr = nn.Linear(self.context_emb_size * self.num_contexts, self.n_layers * self.hidden_size)  # used to initialize rnn in the decoder
        self.tanh = nn.Tanh()

    def forward(self, input):  # input size: (batch, num_contexts)
        embs = [] # (num_contexts, batch, context_emb_size) = (A,B,K)
        embs.append(self.u_embedding(input[:, 0]))  # (batch) --> (batch, context_emb_size)
        embs.append(self.i_embedding(input[:, 1]))  # (batch) --> (bacth, context_emb_size)
        embs.append(self.r_embedding(input[:, 2]))  # (batch) --> (bacth, context_emb_size)

        embedded = torch.cat(embs, 1) # (batch, (num_contexts (u,i,r) * context_emb_size) )
        hidden = self.tanh(self.attr(embedded)).contiguous() # (batch, n_layer*hidden_size)
       
        # used as initialized hidden state for decoder
        hidden = hidden.view(-1, self.n_layers, self.hidden_size).transpose(0, 1).contiguous() # (layer, B, hidden)

        # used as the representation of user, item and rating 
        output = torch.stack(embs) # [(B,K), (B,K), (B,K)] list to tensor (A,B,K), default in dim=0
        return output, hidden

# As a part of AspectAttnDecoder
class AttributeAttn(nn.Module):
    def __init__(self, hidden_size, context_emb_size):
        super(AttributeAttn, self).__init__()
        self.hidden_size = hidden_size
        self.context_emb_size = context_emb_size
        self.attn = nn.Linear(self.hidden_size + context_emb_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        '''
            * Input:
                -- hidden: 
                    previous hidden state of the decoder, in shape (N,B,H)
                -- encoder_outputs:
                    encoder outputs from Encoder, in shape (C,B,K) = (# of contexts, batch size, attribute dimension)
            * Output:
                -- 
                attention energies in shape (B,N,C)
        '''
        attr_len = encoder_outputs.size(0) # C
        seq_len = hidden.size(0) # N
        batch_size = encoder_outputs.size(1)

        H = hidden.repeat(attr_len,1,1,1) # [C,N,B,H]
        encoder_outputs = encoder_outputs.repeat(seq_len,1,1,1).transpose(0,1).contiguous() # [N,C,B,K] -> [C,N,B,K]

        attn_energies = F.tanh(self.score(H, encoder_outputs)) # compute attention score [B,N,C]
        return F.softmax(attn_energies, dim=2) # normalize with softmax on C axis

    def score(self, hidden, encoder_outputs): # hidden (C,N,B,H)
        concat = torch.cat([hidden, encoder_outputs], 3)  # => [C,N,B,(H+K)]
        energy = self.attn(concat) # [C,N,B,(H+K)]->[C,N,B,H]
        energy = energy.view(-1, self.hidden_size) # [C*N*B,H]
        v = self.v.unsqueeze(1) # [H,1]
        energy = energy.mm(v) # [C*N*B,H] x [H,1] -> [C*N*B,1]
        att_energies = energy.view(-1, hidden.size(1), hidden.size(2)) # [C,N,B] 
        att_energies = att_energies.transpose(0, 2).contiguous() # [B,N,C]
        return att_energies

class AspectAttnDecoder(nn.Module):
    def __init__(self, t_embedding, emb_size, hidden_size, context_emb_size, output_size, n_layers=1, dropout=0.2):
        super(AspectAttnDecoder, self).__init__()

        # Keep for reference
        self.emb_size = emb_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_emb_size = context_emb_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Define layers
        self.t_embedding = t_embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size + context_emb_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        self.attr_attn = AttributeAttn(hidden_size, context_emb_size)

    def forward(self, input_seq, last_hidden, encoder_out):
        # Note: we run all steps at one pass
        # Get the embedding of all input words
        '''
        * Input:
            -- input_seq:
                topic input for all time steps, in shape (N=seq_len, B)
            -- last_hidden:
                tuple, last hidden state of the decoder, in shape (n_layers, B, H)
            -- encoder_out: 
                encoder outputs from contexts in shape (C=num_contexts, B, E=context_emb_size)
        '''
        
        batch_size = input_seq.size(1) # B
        seq_len = input_seq.size(0) # N
        a_emb = self.t_embedding(input_seq) # [N,B,E]
        a_emb = self.embedding_dropout(a_emb)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(a_emb, last_hidden)  # rnn_output: [N=seq_len, B, H]  hidden: [n_layer, B, H]

        # Calculate attention
        attn_weights = self.attr_attn(rnn_output, encoder_out) # [N,B,H] x [C,B,E] -> [B,N,C]
        a_context_vector = attn_weights.bmm(encoder_out.transpose(0, 1).contiguous()) # [B,N,C] x [B,C,E] -> [B,N,E]
        a_context_vector = a_context_vector.transpose(0, 1).contiguous()        # [B,N,E] -> [N,B,E]

        tanh_input = torch.cat((rnn_output, a_context_vector), 2) # [N,B, H+E]
        tanh_output = F.tanh(self.concat(tanh_input))  # [N,B,H]
        
        output = self.out(tanh_output) # [N,B,W=vsize]

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

