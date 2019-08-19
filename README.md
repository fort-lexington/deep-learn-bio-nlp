# deep-learn-bio-nlp
[Biocreative II] Gene Mention (GM) task using Deep Learning Models.

## Create a GloVe model

### Corpus

```python
from corpus import DocumentCorpus

docs = DocumentCorpus(root = r'/media/ryan/ExtraDrive1/PMC/XML/')
with open('corpus.txt', 'w') as output:
    for doc in docs:
        text = ' '.join(doc)
        output.write("\n")
```
### Vocabulary

```bash
$ ./vocab_count -max-vocab 150000 -min-count 10 < ../../deep-learn-bio-nlp/corpus.txt > ../../deep-learn-bio-nlp/vocab.txt
BUILDING VOCABULARY
Processed 1784578956 tokens.
Counted 10367082 unique words.
Truncating vocabulary at size 150000.
Using vocabulary of size 150000.
```

### Count Cooccurrences

```bash
$ ./cooccur -window-size 10 -vocab-file ../../deep-learn-bio-nlp/vocab.txt < ../../deep-learn-bio-nlp/corpus.txt > ../../deep-learn-bio-nlp/cooccurrences.bin
COUNTING COOCCURRENCES
window size: 10
context: symmetric
max product: 10485784
overflow length: 28521267
Reading vocab from file "../../deep-learn-bio-nlp/vocab.txt"...loaded 150000 words.
Building lookup table...table contains 90775025 elements.
Processed 1784578876 tokens.
Writing cooccurrences to disk.........140 files in total.
Merging cooccurrence files: processed 902142042 lines.
```

### Shuffle

```bash
$ ./shuffle -verbose 0  < ../../deep-learn-bio-nlp/cooccurrences.bin > ../../deep-learn-bio-nlp/cooccurrences.shuf.bin
SHUFFLING COOCCURRENCES
Merging temp files: processed 902142042 lines.
```

### GloVe Training

```bash
$ ../GloVe-1.2/build/glove -input-file cooccurrences.shuf.bin -vocab-file vocab.txt 
TRAINING MODEL
Read 902142042 lines.
Initializing parameters...done.
vector size: 50
vocab size: 150000
x_max: 100.000000
alpha: 0.750000
iter: 001, cost: 0.033901
iter: 002, cost: 0.022650
iter: 003, cost: 0.020678
iter: 004, cost: 0.019880
iter: 005, cost: 0.019444
iter: 006, cost: 0.019165
iter: 007, cost: 0.018970
iter: 008, cost: 0.018824
iter: 009, cost: 0.018710
iter: 010, cost: 0.018618
iter: 011, cost: 0.018541
iter: 012, cost: 0.018476
iter: 013, cost: 0.018420
iter: 014, cost: 0.018371
iter: 015, cost: 0.018327
iter: 016, cost: 0.018289
iter: 017, cost: 0.018254
iter: 018, cost: 0.018221
iter: 019, cost: 0.018193
iter: 020, cost: 0.018166
iter: 021, cost: 0.018141
iter: 022, cost: 0.018118
iter: 023, cost: 0.018097
iter: 024, cost: 0.018077
iter: 025, cost: 0.018058
```
