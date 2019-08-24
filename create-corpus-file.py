from corpus import DocumentCorpus
import logging
import sys

status_inc = 10000

def main(corpus_path, output_text):
    docs = DocumentCorpus(root = corpus_path)

    with open(output_text, 'w') as output:
        for ix, doc in enumerate(docs):
            text = ' '.join(doc)
            output.write(text)
            output.write("\n")
            if ix % status_inc == 0:
                logging.info('documents: {0}'.format(ix))
    logging.info('DONE')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    corpus_text = sys.argv[1]
    main(r'/media/ryan/ExtraDrive1/PMC/XML/', 'corpus.txt')

