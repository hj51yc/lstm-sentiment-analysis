from __future__ import division

import struct
import sys

def load_word2vec_model(filename):
    FILE_NAME = filename
    MAX_VECTORS = 200000 # This script takes a lot of RAM (>2GB for 200K vectors), if you want to use the full 3M embeddings then you probably need to insert the vectors into some kind of database
    FLOAT_SIZE = 4 # 32bit float
    
    vectors = dict()
    vector_len = 0    
    with open(FILE_NAME, 'rb') as f:
        
        c = None
        
        # read the header
        header = ""
        while c != "\n":
            c = f.read(1)
            header += c
    
        total_num_vectors, vector_len = (int(x) for x in header.split())
        num_vectors = min(MAX_VECTORS, total_num_vectors)
        
        print "Number of vectors: %d/%d" % (num_vectors, total_num_vectors)
        print "Vector size: %d" % vector_len
    
        while len(vectors) < num_vectors:
    
            word = ""        
            while True:
                c = f.read(1)
                if c == " ":
                    break
                word += c
    
            binary_vector = f.read(FLOAT_SIZE * vector_len)
            vectors[word.strip()] = [ struct.unpack_from('f', binary_vector, i)[0] 
                              for i in xrange(0, len(binary_vector), FLOAT_SIZE) ]
            
    return vectors, vector_len



#import cPickle
#
#print "\nSaving..."
#with open(FILE_NAME[:-3] + "pcl", 'wb') as f:
#    cPickle.dump(vectors, f, cPickle.HIGHEST_PROTOCOL)
