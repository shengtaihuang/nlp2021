import pandas as pd
import numpy as np
import json, pickle


def get_aspects(dataset):
    '''
        :Get extracted aspects
        :return df_aspects
            :columns (u_id, i_id, rating, aspect-list)
    '''
    pairs = []
    with open('./data/{}/train.json'.format(dataset), 'r') as f:
        for l in f.readlines():
            l = eval(l)
            u_id = l['user']
            i_id = l['item']
            rating = l['rating']-1
            
            # l['sentence']: [(ASPECT, SENTIMENT-WORD, RELATED-SENTENCE, SENTIMENT-SCORE), ...]
            # if no topic extracted, then assign an empty list
            aspects = [ s[0] for s in l['sentence'] ] if 'sentence' in l.keys() else []
            pairs.append([u_id, i_id, rating, aspects])

    df_aspects = pd.DataFrame(pairs, columns=['u_id', 'i_id', 'rating', 'aspects'])
    return df_aspects


def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def pmi(freq_u_i, freq_u, freq_i):
    '''
        :Calculate per-aspect PMI for each user x item
    '''
    
    score = np.divide(freq_u_i, (freq_u * freq_i), out=np.zeros_like(freq_u_i), where=((freq_u!=0)))
    #print('in pmi before: ', score)
    score = np.nan_to_num(score, nan=0)
    #els = np.nonzero(score)
    #print('print non-zeros: ', score[els])
    normalized_score = normalize(score + 1e-16)
    #els = np.nonzero(normalized_score)
    #print('in pmi after norm: ', normalized_score[els])
    
    return normalized_score

def get_pmis(df_per_aspect):
    '''
        :Calculate PMI for all aspects
        :return df_pmi
    '''
    # Get frequency matrix for all users, items, and users x items
    df_feature_freq_U = pd.crosstab(df_per_aspect.u_id, df_per_aspect.aspect)
    df_feature_freq_I = pd.crosstab(df_per_aspect.i_id, df_per_aspect.aspect)
    df_feature_freq_U_I = pd.crosstab([df_per_aspect.i_id, df_per_aspect.u_id], df_per_aspect.aspect)
    
    pmis = []
    for i, ((i_id, u_id), freq_u_i) in enumerate(df_feature_freq_U_I.iterrows()):
        freq_u = df_feature_freq_U.loc[u_id].values.astype('float64') # a freq vector for the user against all aspects
        freq_i = df_feature_freq_I.loc[i_id].values.astype('float64') # a freq vector for the item against all aspects
        freq_u_i = freq_u_i.values.astype('float64') # a freq vector for the (user x item) against all aspects

        pmi_u_i = pmi(freq_u_i, freq_u, freq_i)
        pmis.append(list(pmi_u_i))
    
    return pd.DataFrame(pmis, index=df_feature_freq_U_I.index, columns=df_feature_freq_U_I.columns)