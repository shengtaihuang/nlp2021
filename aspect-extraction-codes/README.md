The toolkit can be obtained from https://github.com/evison/Sentires  and  https://github.com/lileipisces/Sentires-Guide 


1. Make sure that your machine has Java Runtime Environment installed by running `java -version`. If it is installed, it will tell the version of the java, otherwise, it will say ”Java command not found” and you have to install it first from http://www.java.com/en/download/manual.jsp. 
2. The codes in this repository are already the product of the steps following the setup guidance from https://github.com/lileipisces/Sentires-Guide so you can directly use it. If you want to get it from the scratch instead, you can do the following steps:
- Download and extract the tool from https://drive.google.com/file/d/1RMYPsnxNEUPAH8YQ2iyVRHkVKBek0A0z/view?usp=sharing 
- Download the tool helper from https://github.com/lileipisces/Sentires-Guide  and put the lei folder and the `run.sh` inside the `English-Jar` folder. You can do this command in your terminal (make sure the working directory is in the English-Jar folder): git clone https://github.com/lileipisces/Sentires-Guide 
- Move the `lei` folder and `run_lei.sh` from `English-Jar/Sentires-Guide` folder to `English-Jar` folder.
3. Download the dataset from https://jmcauley.ucsd.edu/data/amazon/ (the small set category for electronics) or ​​http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz, extract it, and put it in the folder `English-Jar/lei/input`.
4. Do sampling. In our case, the code for sampling can be seen in our notebook `English-Jar/lei/input/sampling.ipynb`. The fraction that originally we used is 0.21 fraction. However, it takes so many hours to process. Hence, in this ipynb we changed it to be 0.001.
5. Put the JSON file of the sample for example using our case `sample_review_electronics_version2_bigger.json` in the `English-Jar/lei/input`
6. Edit the `0.format.py` in `English-Jar/lei` and the name file for the raw_path with the name of the JSON file for the samples. For example: `raw_path = 'input/sample_review_electronics_version2_bigger.json`. Since the python codes in the original `0.format.py` file is quite old, change the code to be as follows.

```Python
import pickle
import gzip
import re

raw_path = 'input/sample_review_electronics_version2_bigger.json'  # path to load raw reviews
writer_1 = open('input/record.per.row.txt', 'w', encoding='utf-8')
product2text_list = {}
product2json = {}
for line in open(raw_path, 'r'):
    review = eval(line)
    text = ''
    if 'summary' in review:
        summary = review['summary']
        if summary != '':
            text += summary + '\n'
    text += review['reviewText']

    writer_1.write('<DOC>\n{}\n</DOC>\n'.format(text))

    item_id = review['asin']
    json_doc = {'user': review['reviewerID'],
                'item': item_id,
                'rating': int(review['overall']),
                'text': text}

    if item_id in product2json:
        product2json[item_id].append(json_doc)
    else:
        product2json[item_id] = [json_doc]

    if item_id in product2text_list:
        product2text_list[item_id].append('<DOC>\n{}\n</DOC>\n'.format(text))
    else:
        product2text_list[item_id] = ['<DOC>\n{}\n</DOC>\n'.format(text)]

with open('input/records.per.product.txt', 'w', encoding='utf-8') as f:
    for (product, text_list) in product2text_list.items():
        f.write(product + '\t' + str(len(text_list)) + '\tfake_URL')
        text = '\n\t' + re.sub('\n', '\n\t', ''.join(text_list).strip()) + '\n'
        f.write(text)

pickle.dump(product2json, open('input/product2json.pickle', 'wb')

```

7. Open terminal and set the working directory to  English-Jar/lei.
8. Run this code in the terminal: python3 0.format.py
9. Set the working directory to English-Jar
10. Run this code in the terminal: java -jar thuir-sentires.jar -t pre -c lei/1.pre
11. Run this code: java -jar thuir-sentires.jar -t pos -c lei/2.pos
12. Run this code: cp lei/intermediate/pos.1.txt lei/intermediate/pos.2.txt
13. Run this code: java -jar thuir-sentires.jar -t validate -c lei/3.validate
14. Change the “include=/home/comp/csleili/0/….” to your own absolute path. For example, in our case we change these lines:

```bash
include=/home/comp/csleili/0/English-Jar/preset/relax.threshold
include=/home/comp/csleili/0/English-Jar/preset/english.resource
include=/home/comp/csleili/0/English-Jar/preset/english.pattern
include=/home/comp/csleili/0/English-Jar/preset/default.mapping

# to be:

include=/home/nefriana/English-Jar/preset/relax.threshold
include=/home/nefriana/English-Jar/preset/english.resource
include=/home/nefriana/English-Jar/preset/english.pattern
include=/home/nefriana/English-Jar/preset/default.mapping

```

15. Run this code: `java -jar thuir-sentires.jar -t lexicon -c lei/4.lexicon.linux`
16. Run this code: `java -jar thuir-sentires.jar -t profile -c lei/5.profile`
17. Run this code: `python3 lei/6.transform.py`
18. Run this code: `python3 lei/7.match.py`
19. You can find the extracted aspects in folder `English-Jar/lei/output` with the name file `reviews.pickle`

