import json
import spacy
from collections import Counter, defaultdict
import spacy as nlp
import logging

class BC2Reader(object):

    def __init__(self, train_in, gene_eval):
        self.train_in = train_in
        self.gene_eval = gene_eval
        self.mentions = defaultdict(list)
        self.nlplib = spacy.load('en')
        self.vocab = Counter()
        self._load_eval()

    def _load_eval(self):
        with open(self.gene_eval, 'r') as mentions_fh:
            logging.info('reading eval file')
            for line in mentions_fh:
                sent_id, bounds, text = line[:-1].split('|')
                i, j = bounds.split(' ')
                self.mentions[sent_id].append((int(i), int(j), text))

    def convert(self, fh, format='bio'):
        formatted = list()
        with open(self.train_in, 'r') as train_fh:
            logging.info('tokenizing sentences')
            for line in train_fh:
                sent_id = line[:14]
                text = line[15:-1]
                token_bounds = self._offset_format(text)
                tokens = [x[0] for x in token_bounds]
                self.vocab.update(tokens)
                labels = self.convert_bio(sent_id, token_bounds)
                # formatted.append((sent_id, list(zip(tokens, labels))))
                formatted.append((sent_id, tokens, labels))

        with open(fh, 'w') as json_fh:
            logging.info('writing json output')
            json.dump(formatted, json_fh, indent=2)

    def _offset_format(self, text):
        tokens = [token.text for token in self.nlplib(text)]
        token_bounds = list()
        running_count = 0
        for tok in tokens:
            token_bounds.append((tok, running_count, running_count + len(tok) - 1))
            running_count += len(tok)
        return token_bounds

    def convert_bio(self, sent_id, sent_bounds):
        labels = ['O'] * len(sent_bounds)
        for mention in self.mentions[sent_id]:
            start = mention[0]
            end = mention[1]
            for ix, bounds in enumerate(sent_bounds):
                if bounds[1] == start:
                    labels[ix] = 'B'
                elif bounds[1] > start and bounds[2] <= end:
                    labels[ix] = 'I'
        return labels

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    home = '/home/ryan/Development/deep-learn-bio-nlp/bc2/bc2geneMention/train'
    reader = BC2Reader('{0}/train.in'.format(home),'{0}/GENE.eval'.format(home))
    reader.convert('{0}/converted.json'.format(home))