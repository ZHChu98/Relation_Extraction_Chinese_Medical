import numpy as np
import os
import math
import string
import random


#   0-->无定义关系
#   1--><e1>导致<e2>
#   2--><e1>不导致<e2> 
#   3--><e1>检查发现<e2>
#   4--><e1>执行检查<e2>
#   5--><e1>调查历史<e2>
#   6--><e1>并发症<e2>
#   7--><e1>指代为<e2>
#   8--><e2>导致<e1>
#   9--><e2>不导致<e1>
#  10--><e2>检查发现<e1>
#  11--><e2>执行检查<e1>
#  12--><e2>调查历史<e1>
#  13--><e2>并发症<e1>
#  14--><e2>指代为<e1>

    
def read_data_sets(data_dir, shuffle=True, padding=False, noZero=False, simple=False):
    if simple == False and os.path.exists('word2vec500.npy') == False:
        print("please train word vectors first")
        return
    elif simple==True and os.path.exists('word2vec500_simple.npy')==False:
        print("please train simple word vectors first")
        return
    
    data_list = os.listdir(data_dir)
    sentences = []
    labels = []
    postags = []
    if simple==False:
        count = 0
        for i in range(len(data_list)):
            fp = open(os.path.join('data', data_list[i]), 'r', encoding='utf-8')
            sentence = fp.readline()
            while sentence:
                sentence = sentence.split(' ')
                sentence.remove('\n')

                fp.readline()
                postag = fp.readline()
                postag = postag.split(' ')
                postag.remove('\n')

                label = fp.readline()
                label = label.rstrip('\n')
                if noZero==True and label=='0':
                    sentence = fp.readline()
                    continue
                else:
                    label = int(label) - noZero
                sentences.append(sentence)
                labels.append(label)
                postags.append(postag)
                sentence = fp.readline()
            fp.close
            count += 1
            if count%10 == 0:
                print('#', end='')
    else:
        count = 0
        for i in range(len(data_list)):
            fp = open(os.path.join('data', data_list[i]), 'r', encoding='utf-8')
            fp.readline()
            sentence = fp.readline()
            while sentence:
                sentence = sentence.split(' ')
                sentence.remove('\n')
                
                postag = fp.readline()
                postag = postag.split(' ')
                postag.remove('\n')
                
                label = fp.readline()
                label = label.rstrip('\n')
                if noZero==True and label=='0':
                    fp.readline()
                    sentence = fp.readline()
                    continue
                else:
                    label = int(label) - noZero
                sentences.append(sentence)
                labels.append(label)
                postags.append(postag)
                fp.readline()
                sentence = fp.readline()
            fp.close
            count += 1
            if count%10 == 0:
                print('#', end='')
        
    corpus_size = len(labels)
    if shuffle==True:
        rand_seed = 24 
        # rand_seed = random.randint(0, 100)
        random.seed(rand_seed)
        random.shuffle(sentences)
        random.seed(rand_seed)
        random.shuffle(labels)
        random.seed(rand_seed)
        random.shuffle(postags)

    labels = np.array(labels)
    test_size = corpus_size // 5
    semdata = SemData(sentences, postags, labels, test_size, padding, noZero, simple)
    distribution(labels, test_size, noZero)
    return semdata

    
class Data(object):
    def __init__(self, sentences, postags, labels, embedding_dict, postag_dict, padding, noZero):
        max_length = 110
        n_word = len(embedding_dict)
        n_tag = len(postag_dict)
        self.pos = 0
        self.sentences = []
        self.postags = []
        self.labels = None

        for i in range(len(labels)):
            sentence = [embedding_dict[word] for word in sentences[i]]
            postag = [postag_dict[tag] for tag in postags[i]]
            if padding==True:
                for _ in range(max_length - len(sentence)):
                    sentence.append(n_word)
                    postag.append(n_tag)
            self.sentences.append(sentence)
            self.postags.append(postag)
            
        #self.sentences = np.array(self.sentences)
        if noZero==True:
            self.labels = np.zeros([len(labels), 14], dtype=int)
        else:
            self.labels = np.zeros([len(labels), 15], dtype=int)
        for i in range(len(labels)):
            self.labels[i, labels[i]] = 1
        
    def next_batch(self, batch_size):
        if self.pos+batch_size > len(self.sentences) or self.pos+batch_size > len(self.labels):
            self.pos = 0
        res = (self.sentences[self.pos:self.pos+batch_size], self.postags[self.pos:self.pos+batch_size], 
               self.labels[self.pos:self.pos+batch_size])
        self.pos += batch_size
        return res
    
    
class SemData(object):
    def __init__(self, sentences, postags, labels, test_size, padding, noZero, simple):
        n_class = 15
        if simple == False:
            embedding = np.load('word2vec500.npy').item()
        else:
            embedding = np.load('word2vec500_simple.npy').item()
        tmp = list(embedding.values())
        tmp.append(np.zeros(len(tmp[0])))
        self.embedding_matrix = np.array(tmp)
        embedding_list = list(embedding.keys())
        embedding_dict = dict()
        i = 0
        for word in embedding_list:
            embedding_dict[word] = i
            i += 1
            
        postag = np.load('pos2vec500.npy').item()
        tmp = list(postag.values())
        tmp.append(np.zeros(len(tmp[0])))
        self.postag_matrix = np.array(tmp)
        postag_list = list(postag.keys())
        postag_dict = dict()
        i = 0
        for tag in postag_list:
            postag_dict[tag] = i
            i += 1
        
        self.test = Data(sentences[0:test_size], postags[0:test_size], labels[0:test_size], 
                         embedding_dict, postag_dict, padding, noZero)
        self.validation = Data(sentences[test_size:2*test_size], postags[test_size:2*test_size], labels[test_size:2*test_size], 
                               embedding_dict, postag_dict, padding, noZero)
        self.train = Data(sentences[2*test_size:], postags[2*test_size:], labels[2*test_size:], 
                          embedding_dict, postag_dict, padding, noZero)
        self.weight = np.array([len(labels)/sum(labels==i) for i in range(n_class-noZero)])
    

def distribution(labels, test_size, noZero):
    fp = open('distribution.txt', 'w')
    n_class = 15
    
    dist = [sum(labels==i) for i in range(n_class-noZero)]
    fp.write('all data\n')
    for i in range(n_class-noZero):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/len(labels)))
        
    dist = [sum(labels[0:test_size]==i) for i in range(n_class-noZero)]
    fp.write('\ntest data\n')
    for i in range(n_class-noZero):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/test_size))
        
    dist = [sum(labels[test_size:2*test_size]==i) for i in range(n_class-noZero)]
    fp.write('\nvalidation data\n')
    for i in range(n_class-noZero):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/test_size))
        
    dist = [sum(labels[2*test_size:]==i) for i in range(n_class-noZero)]
    fp.write('\ntrain data\n')
    for i in range(n_class-noZero):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/(len(labels)-2*test_size)))
        
    fp.close()