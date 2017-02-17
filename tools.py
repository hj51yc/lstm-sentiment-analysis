import sys, os
from imdb import prepare_data, load_data
from word2vec_binary2text import load_word2vec_model

def dump_imdb_to_word2vec_corpus(filename, n_words=100000):
    train_data, valid_data, test_data = load_data(n_words=n_words)
    with open(filename, 'w') as fp:
        for data in [train_data, valid_data, test_data]:
            x_data, x_label = train_data
            for x in x_data:
                fp.write(' '.join([str(id) for id in list(x)]) + '\n')
        

def change_word2vec_bin2text(bin_file, text_file):
    word2vec, _  = load_word2vec_model(bin_file)
    with open(text_file, 'w') as fp:
        for word in word2vec:
            line = str(word)+ '\t' + ",".join([str(e) for e in word2vec[word]])
            fp.write(line+'\n')




if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'imdb2words':
        dump_imdb_to_word2vec_corpus(sys.argv[2])
    if cmd == "word2vec_bin2text":
        change_word2vec_bin2text(sys.argv[2], sys.argv[3])
    print 'finished'
  


        

