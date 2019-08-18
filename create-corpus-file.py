from corpus import DocumentCorpus

docs = DocumentCorpus(root = r'/media/ryan/ExtraDrive1/PMC/XML/')

with open('corpus.txt', 'w') as output:
    for doc in docs:
        text = ' '.join(doc)
        output.write(text)
        output.write("\n")
