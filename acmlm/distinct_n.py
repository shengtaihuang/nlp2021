# Borrow from https://blog.csdn.net/weixin_38224810/article/details/120185846
import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--gold",
                    default='metrics/ref-raw.txt',
                    type=str,
                    help="The gold corpus.")

parser.add_argument("--pred",
                    default='metrics/tst-clean.txt',
                    type=str,
                    help="The predicted corpus.")

args = parser.parse_args()

gold_file = '{}'.format(args.gold)
pred_file = '{}'.format(args.pred)

def get_dict(tokens, ngram=1, gdict=None):
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None:
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    
    return token_dict

def calc_distinct_ngram(pair_list, ngram=1):
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total +=freq
        ngram_distinct_count += 1
    
    return ngram_distinct_count / ngram_total

sents = []

with open(gold_file, 'r') as g, open(pred_file, 'r') as p:
    for gl, pl in zip(g, p):
        gold_tokens = gl.split(" ")
        pred_tokens = pl.split(" ")
        sents.append([pred_tokens, gold_tokens])

distinct1 = calc_distinct_ngram(pair_list=sents, ngram=1)
distinct2 = calc_distinct_ngram(pair_list=sents, ngram=2)

print("Distinct-1/Distinct-2 = {}/{}".format(round(distinct1, ndigits=4), round(distinct2, ndigits=4)))