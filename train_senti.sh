#!/bin/bash
set -e

words_file="imdb.data.words"
vectors_file="imdb.data.vec"
VECTOR_DIM="64"

WORD2VEC="/home/huangjin/third_tools/word2vec/bin/word2vec"

echo "start to change imdb to text words"
python tools.py imdb2words $words_file

echo "start to train word2vec ..."
$WORD2VEC -train $words_file -output $vectors_file -cbow 0 -size $VECTOR_DIM  -window 2 -negative 0 -hs 1 -sample 1e-4 -threads 20 -binary 1 -iter 1000 -binary 0 

echo "start to create one word2vec text file ..."

python tools.py word2vec_bin2text $vectors_file $vectors_file".txt"

echo "start to train_senti.py"
python train_senti.py $vectors_file

echo "finish"

