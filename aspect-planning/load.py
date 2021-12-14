import pandas as pd
import torch
import json, pickle, re, os

SOS_ID = 0
EOS_ID = 1
PAD_ID = 2

class Vocabulary:
    def __init__(self, name, save_dir):
        self.name = name
        with open(os.path.join(save_dir, 'aspect2idx.pkl'), 'rb') as fp:
            self.aspect2idx = pickle.load(fp)
        with open(os.path.join(save_dir, 'idx2aspect.pkl'), 'rb') as fp:
            self.idx2aspect = pickle.load(fp)
        self.n_topics = len(self.aspect2idx)

def tokenize(path, dataset, vocab, save_dir):
    print("Reading {}".format(path))
    # combine attributes and reviews into pairs
    with open(os.path.join(save_dir, 'user2idx.pkl'), 'rb') as fp:
        user2idx = pickle.load(fp)
    with open(os.path.join(save_dir, 'item2idx.pkl'), 'rb') as fp:
        item2idx = pickle.load(fp)
    
    pairs = []
    with open(path, 'r') as f:
        for l in f.readlines():
            l = eval(l)
            
            # id to idx
            u_idx = user2idx[l['user']] 
            i_idx = item2idx[l['item']]
            rating = l['rating']-1
            
            # l['sentence']: [(ASPECT, SENTIMENT-WORD, RELATED-SENTENCE, SENTIMENT-SCORE), ...]
            # if no topic extracted, then assign an empty list
            aspects = [ s[0] for s in l['sentence'] ] if 'sentence' in l.keys() else []
            aspect_seq = ['<sos>'] + aspects + ['<eos>']
            
            contexts = [u_idx, i_idx, rating]
            pairs.append([contexts, aspect_seq])

    return pairs

def prepareData(vocab, dataset, save_dir):
    print('vocab, dataset, save_dir: ', vocab, dataset, save_dir)
    train_pairs = tokenize(os.path.join(save_dir, 'train.json'), dataset, vocab, save_dir)
    valid_pairs = tokenize(os.path.join(save_dir, 'validation.json'), dataset, vocab, save_dir)
    test_pairs = tokenize(os.path.join(save_dir, 'test.json'), dataset, vocab, save_dir)

    pd.DataFrame(train_pairs, columns=['context', 'aspects']).to_csv(os.path.join(save_dir, 'paired_train_set.csv'))
    pd.DataFrame(valid_pairs, columns=['context', 'aspects']).to_csv(os.path.join(save_dir, 'paired_validation_set.csv'))
    pd.DataFrame(test_pairs, columns=['context', 'aspects']).to_csv(os.path.join(save_dir, 'paired_test_set.csv'))
    
    torch.save(train_pairs, os.path.join(save_dir, 'paired_train.tar'))
    torch.save(valid_pairs, os.path.join(save_dir, 'paired_validation.tar'))
    torch.save(test_pairs, os.path.join(save_dir, 'paired_test.tar'))
    
    return train_pairs, valid_pairs, test_pairs


def loadPrepareData(dataset, save_dir):
    try:
        print("Start loading training data ...")
        vocab = Vocabulary(dataset, save_dir)
        train_pairs = torch.load(os.path.join(save_dir, 'paired_train.tar'))
        valid_pairs = torch.load(os.path.join(save_dir, 'paired_validation.tar'))
        test_pairs = torch.load(os.path.join(save_dir, 'paired_test.tar'))
        print('# train instances: ', len(train_pairs))
        
    except FileNotFoundError:
        print("Saved data not found, start preparing training data ...")
        vocab = Vocabulary(dataset, save_dir)
        train_pairs, valid_pairs, test_pairs = prepareData(vocab, dataset, save_dir)
        
    return vocab, train_pairs, valid_pairs, test_pairs


