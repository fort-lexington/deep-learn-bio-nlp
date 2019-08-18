import lxml.etree as ET
import os
import sys
import logging
import pickle
import math
from gensim import models
from textblob import TextBlob
from collections import defaultdict, Counter

stopwords = set()
with open('stopwords.txt', 'r') as stops:
    for line in stops:
        if len(line.strip()) > 0:
            stopwords.add(line.strip())


def log_likelihood_ratio(k1, n1, k2, n2):
    ''' Note: llr is -2log(lambda), which is roughly chi-squared > 10.8 '''
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    llr = 2 * (logL(p1, k1, n1) + logL(p2, k2, n2) - logL(p, k1, n1) - logL(p, k2, n2))
    return llr

def logL(p,k,n):
    return k * math.log10(p) + (n - k) * math.log10(1 - p)


class Corpus(object):
    ''' Streamable corpus object (only reads one file into memory at a time). '''

    def __init__(self, root=r'/media/ryan/ExtraDrive1/PMC/XML/'):
        self.path_list = list()
        for dir_path, _, file_names in os.walk(root):
            for file in [f for f in file_names if f.endswith('.nxml')]:
                self.path_list.append(os.path.join(dir_path, file))

    def file_count(self):
        return len(self.path_list)

    def __iter__(self):
        ''' Yield one sentence at a time (as a list of tokens) '''
        for path in self.path_list:
            logging.info(path)
            art_text = extract_article(path).get_text()
            for sent in TextBlob(art_text).sentences:
                yield [tok for tok in sent.words] # sent.tokens includes punctuation

class DocumentCorpus(object):
    ''' Streamable corpus object (only reads one file into memory at a time). '''

    def __init__(self, root=r'/media/ryan/ExtraDrive1/PMC/XML/'):
        self.path_list = list()
        for dir_path, _, file_names in os.walk(root):
            for file in [f for f in file_names if f.endswith('.nxml')]:
                self.path_list.append(os.path.join(dir_path, file))

    def file_count(self):
        return len(self.path_list)

    def __iter__(self):
        ''' Yield one document at a time (as a list of tokens) '''
        for path in self.path_list:
            logging.info(path)
            art_text = extract_article(path).get_text()
            art_tokens = []
            for sent in TextBlob(art_text).sentences:
                art_tokens.extend(tok for tok in sent.words)
            yield(art_tokens)

class Article(object):
    ''' Container for text extracted from *.nxml files. See 'extract_article'. '''

    def __init__(self, title, abstract, body):
        self.title = title
        self.abstract = abstract
        self.body = body

        self.title = self.normalize(self.title)
        self.abstract = self.normalize(self.abstract)
        self.body = self.normalize(self.body)

    def get_title(self):
        return self.title

    def get_abstract(self):
        return self.abstract

    def get_body(self):
        return self.body

    def normalize(self, text):
        return text

    def get_text(self):
        text = "\n".join([self.get_title(), self.get_abstract(), self.get_body()])
        return text


def extract_article(path):
    ''' Accepts a file system path for an *.nxml file, parses the XML (removing tables and xref elements), and
    returns an Article object.'''

    with open(path, 'r', encoding='utf8') as xmlfh:
        tree = ET.parse(xmlfh)
    root = tree.getroot()

    title_node = root.find(r'.//article-title')
    abstract_node = root.find(r'.//abstract')
    body_node = root.find(r'.//body')

    # Remove Tables
    tables = root.findall(r'.//table')
    if tables is not None:
        for tab in tables:
            tab.getparent().remove(tab)

    xrefs = root.findall(r'.//xref')
    if xrefs is not None:
        for xref in xrefs:
            xref.getparent().remove(xref)

    title_text = ' ' if title_node is None else r' '.join([x for x in title_node.itertext()])
    abstract_text = ' ' if abstract_node is None else r' '.join([x for x in abstract_node.itertext()])
    body_text = ' ' if body_node is None else r' '.join([x for x in body_node.itertext()])

    article = Article(
        ''.join(title_text),
        ''.join(abstract_text),
        ''.join(body_text))

    return article


class Background(object):
    ''' Simple term counts in the background corpus.
    In order to get the count of all tokens in the corpus, use the key '_total_'.
    '''

    def __init__(self):
        self.counts = defaultdict(lambda: 0)

    def add_all(self, token_list):
        for tok in token_list:
            if tok not in stopwords:
                self.add(tok)

    def get_count(self, term):
        return self.counts.get(term, 0)

    def size(self):
        return self.counts.get('_total_')

    def unique_size(self):
        return len(self.counts.keys())

    def add(self, token):
        self.counts[token] += 1
        self.counts['_total_'] += 1

    def save(self, fname):
        with open(fname, 'wb') as handle:
            pickle.dump(dict(self.counts), handle)

    def load(self, fname):
        with open(fname, 'rb') as handle:
            self.counts = defaultdict(lambda:0, pickle.load(handle))


class Models(object):

    def __init__(self):
        self.bigram = None
        self.background = None

    def train(self, corp):
        print("Files: {0}".format(corp.file_count()))

        logging.info("Build phraser.")

        # Training
        if not os.path.exists('./bigram.model'):
            phrases = models.phrases.Phrases(corp)
            self.bigram = models.phrases.Phraser(phrases)
            self.bigram.save('bigram.model')
        else:
            self.bigram = models.phrases.Phraser.load('./bigram.model')

        logging.info("Build background corpus.")

        self.background = Background()
        if not os.path.exists('./back.pickle'):
            for tokens in corp:
                self.background.add_all(self.bigram[tokens])
            self.get_background().save('back.pickle')
        else:
            self.get_background().load('back.pickle')

    def get_background(self):
        return self.background

    def get_bigram(self):
        return self.bigram

def main():
    ''' Run a demo keyphrase analysis on article. '''
    
    logging.basicConfig(level=logging.INFO)

    test_article = r'/media/ryan/ExtraDrive1/PMC/XML/non_comm_use.A-B.xml/Addict_Health/PMC4137445.nxml'

    corp = Corpus()
    my_models = Models()
    my_models.train(corp)

    # Print an example of collocation detection using the bigram "phraser" model.
    sample = '''In this study, we investigate the role of MIF, MIF-2, sCD74,
    and MIF genotypes in patients scheduled for elective single or complex surgical procedures such as coronary artery bypass grafting or valve replacement.'''

    sample_toks = [tok for sent in TextBlob(sample).sentences for tok in sent.tokens]
    logging.debug(my_models.get_bigram()[sample_toks])

    threshold = 10.8
    article = extract_article(test_article)

    body = TextBlob(article.get_abstract() + "\n" + article.get_body())

    logging.info(article.get_title())
    title_tokens =  [tok for tok in TextBlob(article.get_title()).tokens if tok not in stopwords]
    title_tokens = my_models.get_bigram()[title_tokens]

    doc_tokens = [tok for sent in body.sentences for tok in my_models.get_bigram()[sent.tokens] if tok not in stopwords]
    doc_count = Counter(doc_tokens)

    # Lets find the signature terms
    scored = set()
    signature_terms = set()
    for token in doc_tokens:
        k1 = doc_count[token]
        n1 = len(doc_tokens)
        k2 = my_models.get_background().get_count(token)
        n2 = my_models.get_background().size()

        debug = "{0} {1} {2} {3}".format(k1, n1, k2, n2)
        logging.debug(debug)

        r = log_likelihood_ratio(k1, n1, k2, n2)
        if r > threshold:
            scored.add((r, token))
            signature_terms.add(token)

    # Sort and print the top signature terms
    items = sorted(list(scored), key=lambda term: term[0], reverse=True)
    logging.info("{0} terms.".format(len(items)))
    for s in items[:]:
        print("{0:0.2f}\t{1}".format(s[0], s[1]))

    # Sort and print the topic sentences
    scored_sentences = list()
    for ix, sent in enumerate(body.sentences):
        sentence_tokens = [tok for tok in my_models.get_bigram()[sent.tokens] if tok not in stopwords]
        score = sum([1 for tok in sentence_tokens if tok in signature_terms])
        scored_sentences.append((score, sent.raw, ix+1))
    sorted_scored_sentences = sorted(scored_sentences, key=lambda t: t[0], reverse=True)
    for s in sorted_scored_sentences[:3]:
        print("{0} ({1}): {2}".format(s[2], s[0], s[1]))

    logging.info("Corpora size: %s" % my_models.get_background().size())
    logging.info("Unique tokens: %s" % my_models.get_background().unique_size())

if __name__ == '__main__':
    main()
