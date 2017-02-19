#encoding: utf-8
'''
@author: huangjin (Jeff)
@email: hj51yc@gmail.com
Using one LSTM layer, one sum pooling layer, and one output layer
'''

import sys, os
import numpy as np
np.seterr(all='raise')

def sigmoid(a):
    return 1.0/(1+np.exp(-a))

def dsigmoid(a):
    return a*(1-a)

def tanh(a):
    return np.tanh(a)

def dtanh(a):
    return 1 - (a ** 2)

def softmax(y):
    p = np.exp(y)
    s = np.sum(p)
    final = p / s
    return final


def cross_entropy(prob, y_true):
    log_prob_neg = np.log(1 - prob)
    log_prob = np.log(prob)
    y_true_neg = 1 - y_true
    return -(np.sum(log_prob_neg * y_true_neg) + np.sum(log_prob * y_true))


##simple LSTM: N blocks with one cell in every block!
class LSTM(object):

    def __init__(self, x_dim, hidden_num, output_dim, eta, epsilon, verbose=False):
        self.eta = eta
        self.epsilon = epsilon
        self.adagrads_sum = {}
        
        self.verbose = verbose

        ## concate input as [h, x]
        Z = x_dim + hidden_num
        H = hidden_num
        D = output_dim
        self.Z = Z
        self.H = H
        self.D = D
        self.Wi = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bi = np.zeros((1, H))
        self.Wf = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bf = np.zeros((1, H))
        self.Wo = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bo = np.zeros((1, H))
        self.Wc = np.random.randn(Z, H) / np.sqrt(Z / 2.)
        self.bc = np.zeros((1, H))

        self.Wy = np.random.randn(H, D) / np.sqrt(D / 2.0)
        self.by = np.zeros((1, D))

    
    def predict(self, x_seq_start, state_init):
        state = state_init
        h_array = np.zeros((x_seq_start.shape[0], self.H))
        i = 0
        for x in x_seq_start:
            state, cache = self.forward(x, state)
            h, c = state
            h_array[i] = h
            i += 1
        prob, _ = self.layer_sigmoid(h_array) 
        return prob


    def forward(self, x, state):
        h_prev, c_prev = state
        
        #X = np.column_stack((h_prev, x))
        #print 'x:', x
        #print 'h_prev:', h_prev
        X = np.hstack((h_prev, x.reshape(1, len(x))))
       
        hi = sigmoid(np.dot(X, self.Wi) + self.bi)
        hf = sigmoid(np.dot(X, self.Wf) + self.bf)
        ho = sigmoid(np.dot(X, self.Wo) + self.bo)
        
        hc = tanh(np.dot(X, self.Wc) + self.bc)
        c = hf * c_prev + hi * hc
        h = ho * tanh(c)
        cache = (hi, hf, ho, hc, h, c, c_prev, h_prev, X)
        state = (h, c)
        return state, cache

    def layer_sigmoid(self, h_array):
        h_count = h_array.shape[0]
        h_avg = np.sum(h_array, axis=0) / h_array.shape[0]
        y = np.dot(h_avg, self.Wy) + self.by
        prob = softmax(y)
        info = (h_avg, h_count)
        return prob, info

    def layer_sigmoid_bp(self, prob, y_label, info):
        h_avg, h_count = info
        h_avg = h_avg.reshape((1, len(h_avg)))
        y_index = np.argmax(y_label)
        dy = np.array(prob.copy())
        dy[0, y_index] -= 1
        
        dWy = np.dot(h_avg.T, dy)
        dby = dy

        dh_avg = np.dot(dy, self.Wy.T)
        dh_avg *= 1.0 / h_count
        return dh_avg, dict(Wy=dWy, by=dby)


    def grad_clip(self, dw, rescale=5.0):
        norm = np.sum(np.abs(dw))
        if norm > rescale:
            return dw * (rescale / norm)
        else:
            return dw

    def backward(self,d_next, dh_avg, cache):
        hi, hf, ho, hc, h, c, c_prev, h_prev, X = cache
        dh_next, dc_next = d_next

        # Note we're adding dh_next here, because h is forward in next_step and make output y here: h is splited here!
        dh = dh_avg +  dh_next
        if self.verbose:
            print 'the dh', dh
        
        dho = tanh(c) * dh
        dho = dsigmoid(ho) * dho
        if self.verbose:
            print 'the dho', dho
        
        # Gradient for c in h = ho * tanh(c), note we're adding dc_next here! 
        #dc = ho * dh + dc_next
        #dc = dtanh(c) * dc
        
        ## i change dc below
        dc = dh * ho * dtanh(c) + dc_next
        if self.verbose:
            print 'the dc', dc

        dhc = hi * dc
        dhc = dhc * dtanh(hc)
        if self.verbose:
            print 'the dhc', dhc

        dhf = c_prev * dc
        dhf = dsigmoid(hf) * dhf
        if self.verbose:
            print 'the dhf', dhf

        dhi = hc * dc
        dhi = dsigmoid(hi) * dhi
        if self.verbose:
            print 'the dhi', dhi

        dWf = np.dot(X.T , dhf)
        dbf = dhf
        dXf = np.dot(dhf, self.Wf.T)
        if self.verbose:
            print 'the X.T', X.T
            print 'the dWf', dWf
            print 'the dbf', dbf

        dWi = np.dot(X.T, dhi)
        dbi = dhi
        dXi = np.dot(dhi, self.Wi.T)

        dWo = np.dot(X.T, dho)
        dbo = dho
        dXo = np.dot(dho, self.Wo.T)

        dWc = np.dot(X.T, dhc)
        dbc = dhc
        dXc = np.dot(dbc, self.Wc.T)

        dX = dXf + dXi + dXo + dXc
        new_dh_next = dX[:, :self.H]
        new_dh_next = self.grad_clip(new_dh_next)
        if self.verbose:
            print "the dh_next", new_dh_next

        # Gradient for c_old in c = hf * c_old + hi * hc
        new_dc_next = hf * dc
        new_dc_next = self.grad_clip(new_dc_next)
        if self.verbose:
            print 'the dc_next', new_dc_next

        grad = dict(Wf=dWf, Wi=dWi, Wo=dWo, Wc=dWc, bf=dbf, bi=dbi, bc=dbc, bo=dbo)
        for key in grad:
            grad[key] = self.grad_clip(grad[key])
        new_d_next = (new_dh_next, new_dc_next) 

        return new_d_next, grad

    

    def train_step(self, x_seq, y_label, state):
        grads = {}
        probs = []
        caches = []
        loss = 0.0
        h, c = state
        h_array = np.zeros((x_seq.shape[0], self.H))
        i = 0
        for x in x_seq:
            state, cache = self.forward(x, state)
            caches.append(cache)
            h, c = state
            h_array[i] = h
            i += 1

        prob, info = self.layer_sigmoid(h_array)
        
        if self.verbose:
            print 'prob', prob, 'y_label', y_label

        dh_avg, cur_grads = self.layer_sigmoid_bp(prob, y_label, info)
        grads = cur_grads
        
        loss = cross_entropy(prob, y_label)
        if self.verbose:
            print 'loss', loss 
        d_next = (np.zeros_like(h), np.zeros_like(c))

        for cache in reversed(caches):
            d_next, cur_grads = self.backward(d_next, dh_avg, cache)
            for key in cur_grads:
                if key not in grads:
                    grads[key] = cur_grads[key]
                else:
                    grads[key] += cur_grads[key]

        return grads, loss, state

    
    def adagrad(self, grads, eta, epsilon):
        
        for w_name in grads:
            W = getattr(self, w_name)
            dW = grads[w_name]
            try:
                square_dW = dW * dW
            except:
                print 'overflow dW', w_name
                print 'overflow dW', dW
                for t in grads:
                    print 'w', t, getattr(self, t)
                    print 'dw', t, grads[t]
                raise
            if w_name in self.adagrads_sum:
                self.adagrads_sum[w_name] += square_dW
            else:
                self.adagrads_sum[w_name] = square_dW + epsilon

            W_step =  eta / np.sqrt(self.adagrads_sum[w_name])
            W -= W_step * dW


    def train_once(self, x_seq, y_label, state):
        grads, loss, state = self.train_step(x_seq, y_label, state)
        self.adagrad(grads, self.eta, self.epsilon)
        return loss, state

