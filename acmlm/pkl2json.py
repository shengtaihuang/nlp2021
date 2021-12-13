import json
import pickle as pkl
import re

with open('amazon_electronics_reviews.pickle', 'rb') as pickle_file:
    reviews = pkl.load(pickle_file)
    
aspect = []

# Only keep aspects with positive sentiment
for r in reviews:
    if dict.get(r, "sentence") != None:
        for s in r['sentence']:
            if s[-1] == 1:
                aspect += s[0:2]

# Compute frequency of aspects
freq = {}
for item in aspect:
    if (item in freq):
        freq[item] += 1
    else:
        freq[item] = 1

# Keep aspects occur >= 5
aspect_gt5 = [k for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=False) if v >= 5]

# Split sentence into two elements, 'fine_aspect' and 'sentence'
rev2 = []
for r in reviews:
    if dict.get(r, "sentence") != None:
        for j in range(len(r['sentence'])):
            if r['sentence'][j][-1] == 1:
                if r['sentence'][j][0] in aspect_gt5 and r['sentence'][j][1] in aspect_gt5:
                    rev2.append({'item': r['item'],
                                 'rating': r['rating'],
                                 'fine_aspect': list(r['sentence'][j][0:2]),
                                 'sentence': r['sentence'][j][2],
                                 'text': r['text'],
                                 'user': r['user']})

# Recover extracted sentence fragments to complete sentences
for r in rev2:
    text = re.sub(pattern="[+*\"\[\]]", repl="", string=r['text'])
    text = re.sub(pattern="\\\\", repl=" ", string=text)
    text = re.sub(pattern="\\.{2,}", repl=" ", string=text)
    text = re.sub(pattern=" +\' +", repl="'", string=text)
    text = re.sub(pattern=" +([,.?!])", repl="\\1", string=text)
    text = re.sub(pattern=" +", repl=" ", string=text)
    text = re.sub(pattern="^ +", repl="", string=text)
    text = re.sub(pattern=" +$", repl="", string=text)
    text = re.split(pattern="\\n|\\.[^)]|\\!+[^)]|\\?+[^)]", string=text)
    
    for ele in text:
        pat = re.sub(pattern="[+*\"\[\]]", repl="", string=r['sentence'])
        pat = re.sub(pattern="\\\\", repl=" ", string=pat)
        pat = re.sub(pattern="\\.{2,}", repl=" ", string=pat)
        pat = re.sub(pattern=" +\' +", repl="'", string=pat)
        pat = re.sub(pattern=" +([,.?!])", repl="\\1", string=pat)
        pat = re.sub(pattern=" +", repl=" ", string=pat)
        pat = re.sub(pattern="^ +", repl="", string=pat)
        pat = re.sub(pattern=" +$", repl="", string=pat)
        try:
            k = re.search(pattern=pat, string=ele)
        except:
            break
        if k != None:
            r['sentence'] = k.string
            break

# Combine aspects correspond to the same sentence
sent = []
rev3 = []
for r in rev2:
    if r['sentence'] not in sent:
        sent.append(r['sentence'])
        rev3.append({'item': r['item'],
                     'rating': r['rating'],
                     'fine_aspect': r['fine_aspect'],
                     'sentence': r['sentence'],
                     'text': r['text'],
                     'user': r['user']})
    else:
        which_idx = [r3['sentence'] for r3 in rev3].index(r['sentence'])
        fa = list(set(rev3[which_idx]['fine_aspect'] + r['fine_aspect']))
        rev3[which_idx]['fine_aspect'] = fa

# Save the processed data as json
with open('data/amazon/amazon.json', 'w') as fp:
    for line in rev3:
        json.dump(line, fp)
        fp.write("\n")