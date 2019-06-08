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

    
def read_data_sets(data_dir, padding=0, shuffle=True, noZero=False):
    if os.path.exists('word2vec.npy') == False:
        print("please train word vectors first")
        return
    else:
        data_list = os.listdir(data_dir)
        sentences = []
        labels = []
        for i in range(len(data_list)):
            fp = open(os.path.join('data', data_list[i]), 'r', encoding='utf-8')
            sentence = fp.readline()
            while sentence:
                sentence = sentence.split(' ')
                sentence.remove('\n')
                for _ in range(2):
                    fp.readline();
                label = fp.readline()
                label = label.rstrip('\n')
                if noZero==True and label=='0':
                    sentence = fp.readline()
                    continue
                else:
                    label = int(label) - noZero
                sentences.append(sentence)
                labels.append(label)
                sentence = fp.readline()
            fp.close
        corpus_size = len(labels)
        if shuffle==True:
            rand_seed = random.randint(0, 100)
            random.seed(rand_seed)
            random.shuffle(sentences)
            random.seed(rand_seed)
            random.shuffle(labels)
        
        labels = np.array(labels)
        train_size = corpus_size // 10 * 9
        semdata = SemData(sentences, labels, train_size, padding, noZero)
        distribution(labels, train_size, noZero)
    return semdata

    
class Data(object):
    def __init__(self, sentences, labels, padding, noZero):
        n_embedding = 300
        self.pos = 0
        self.sentences = []
        self.labels = None

        embeddings = np.load("word2vec.npy").item()

        for i in range(len(labels)):
            sentence = [embeddings[word] for word in sentences[i]]
            if not padding==0:
                zero_length = padding - len(sentence[i])
                for i in range(zero_length):
                    sentence.append(np.zeros([n_embedding], float))
            self.sentences.append(np.array(sentence))
            
        self.sentences = np.array(self.sentences)
        if noZero==True:
            self.labels = np.zeros([len(labels), 14], dtype=int)
        else:
            self.labels = np.zeros([len(labels), 15], dtype=int)
        for i in range(len(labels)):
            self.labels[i, labels[i]] = 1
        
    def next_batch(self, batch_size):
        if self.pos+batch_size > len(self.sentences) or self.pos+batch_size > len(self.labels):
            self.pos = 0
        res = (self.sentences[self.pos:self.pos+batch_size], self.labels[self.pos:self.pos+batch_size])
        self.pos += batch_size
        return res
    
    
class SemData(object):
    def __init__(self, sentences, labels, train_size, padding, noZero):
        n_class = 15
        self.train = Data(sentences[0:train_size], labels[0:train_size], padding, noZero)
        self.test = Data(sentences[train_size:], labels[train_size:], padding, noZero)
        self.weight = np.array([len(labels)/sum(labels==i) for i in range(n_class-noZero)])
    

def distribution(labels, train_size, noZero):
    fp = open('distribution.txt', 'w')
    n_class = 15
    
    dist = [sum(labels==i) for i in range(n_class-noZero)]
    fp.write('all data\n')
    for i in range(n_class-noZero):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/len(labels)))
    dist = [sum(labels[0:train_size]==i) for i in range(n_class-noZero)]
    fp.write('\ntrain data\n')
    for i in range(n_class-noZero):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/train_size))
    dist = [sum(labels[train_size:]==i) for i in range(n_class-noZero)]
    fp.write('\ntest data\n')
    for i in range(n_class-noZero):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/(len(labels)-train_size)))
        
    fp.close()