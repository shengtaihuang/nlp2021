import argparse
import re

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--raw",
                    default='metrics/ref-raw.txt',
                    type=str,
                    help="The raw corpus.")

parser.add_argument("--clean",
                    default='metrics/tst-clean.txt',
                    type=str,
                    help="The clean corpus.")

args = parser.parse_args()

raw_file = '{}'.format(args.raw)
clean_file = '{}'.format(args.clean)

sents = []
with open(raw_file, 'r') as r:
    for line in r:
        line = re.sub(pattern="\[CLS\]|\[SEP\]|\[PAD\]|\[unused[0-9]+\]|\\n", repl="", string=line)
        line = re.sub(pattern=" +", repl=" ", string=line)
        line = re.sub(pattern="^ +| +$", repl="", string=line)
        sents.append(line)

with open(clean_file, 'w') as fp:
    for s in sents:
        fp.write(s)
        fp.write('\n')