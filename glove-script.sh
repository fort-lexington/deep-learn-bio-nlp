#!/bin/bash

MODEL_DIR=/home/ryan/Development/deep-learn-bio-nlp
GLOVE_DIR=/home/ryan/Development/GloVe-1.2/build
MAX_VOCAB=100000
MODEL_NAME=vectors_100K
GLOVE_ITER=7
OUTPUT_DIM=256
# Example using mostly GloVe defaults, with a few exceptions

$GLOVE_DIR/vocab_count -max-vocab $MAX_VOCAB -min-count 10 < $MODEL_DIR/corpus.txt > $MODEL_DIR/vocab.txt

$GLOVE_DIR/cooccur -window-size 10 -vocab-file $MODEL_DIR/vocab.txt < $MODEL_DIR/corpus.txt > $MODEL_DIR/cooccurrences.bin

$GLOVE_DIR/shuffle -verbose 0  < $MODEL_DIR/cooccurrences.bin > $MODEL_DIR/cooccurrences.shuf.bin

$GLOVE_DIR/glove -iter $GLOVE_ITER -binary 2 -vector-size 256 -input-file $MODEL_DIR/cooccurrences.shuf.bin -vocab-file $MODEL_DIR/vocab.txt -save-file $MODEL_NAME
