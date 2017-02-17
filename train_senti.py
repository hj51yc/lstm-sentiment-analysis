import sys, os, time, random
import numpy as np

from my_lstm import LSTM, cross_entropy
from imdb import prepare_data, load_data
from word2vec_binary2text import load_word2vec_model

n_words=100000

def load_my_data():
    train_data, valid_data, test_data = load_data(n_words=n_words)
    train_x, train_y = train_data
    print train_x[0], train_y[0]
    return train_data, valid_data, test_data

def make_x_seq_and_x_label(word2vec, seq, label):
    x_seq = None
    for i in xrange(len(seq)):
        vec = word2vec.get(str(seq[i]))
        if not vec:
            continue
        if x_seq is None:
            x_seq = np.zeros((len(seq), len(vec)))
        x_seq[i] = vec
    x_label = np.zeros((1, 2))
    x_label[0][label] = 1.0
    return x_seq, x_label


def my_shuffle(datas):
    return random.shuffle(datas)

def evalute_data(lstm, word2vec, test_datas, state_init):
    test_x_datas, test_x_labels =  test_datas
    loss = 0
    acc = 0
    for (seq, label) in zip(test_x_datas, test_x_labels):
        x_seq, x_label = make_x_seq_and_x_label(word2vec, seq, label)
        prop = lstm.predict(x_seq, state_init)
        loss += cross_entropy(prop, x_label)
        prop = prop[0]
        if label == 0 and prop[0] > prop[1]:
            acc += 1
        if label == 1 and prop[1] > prop[0]:
            acc += 1
    print 'loss:', loss
    print 'acc:', acc * 1.0 / len(test_x_labels)



def train_model(word2vec_model_file):
    print 'loading data ...'
    train_datas, test_datas, test_labels = load_my_data()
    
    print 'load word2vec model ...'
    word2vec, vec_dim = load_word2vec_model(word2vec_model_file)
    print 'word2vec len', len(word2vec)
    print 'key[0]', word2vec.keys()[0]

    x_datas, x_labels = train_datas
    
    x_dim = vec_dim
    hidden_num = 30
    out_dim = 2
    lstm = LSTM(x_dim, hidden_num, out_dim, 0.3, 1.0e-8)

    iter = 1000
    print 'start to train ...'
    now = int(time.time())

    h_init = np.zeros((1, hidden_num))
    c_init = np.zeros((1, hidden_num))
    k = 0
    init_state = (h_init, c_init)
    total_loss = 0
    total_count = 0
    for i in xrange(iter):
        state = (h_init, c_init)
        data = [(seq, label) for seq, label in zip(x_datas, x_labels)]
        print 'data len:', len(data)
        my_shuffle(data)
        print 'begin'
        for (seq, label) in data:
            x_seq, x_label = make_x_seq_and_x_label(word2vec, seq, label)
            loss, state = lstm.train_once(x_seq, x_label, init_state)
            total_count += 1
            total_loss += loss
            ###loss, state = lstm.train_once(x_seq, x_label, state)
            print 'k', k, 'avg_loss', total_loss / total_count
            if k % 100 == 0:
                print 'k', k, 'avg_loss', total_loss / total_count
                print 'cost_time:', int(time.time()) - now
                now = int(time.time())
            k += 1
            if loss < 0.0001:
                break
        
        evalute_data(lstm, word2vec, test_datas, init_state)
    
    #test_gen(lstm, x_test_seq, init_state, stop_indexes) 
    print 'finished'



if __name__ == "__main__":
    word2vec_model_file = sys.argv[1]
    train_model(word2vec_model_file)
