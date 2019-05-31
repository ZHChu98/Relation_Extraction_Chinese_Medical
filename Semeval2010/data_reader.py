import numpy as np
import os
import math
import string
from string import digits

##################################
#               Other :  0       #
#        Cause-Effect :  1, 2    #
#     Component-Whole :  3, 4    #
#  Entity-Destination :  5, 6    #
#    Product-Producer :  7, 8    #
#       Entity-Origin :  9, 10   #
#   Member-Collection :  11, 12  #
#       Message-Topic :  13, 14  #
#   Content-Container :  15, 16  #
#   Instrument-Agency :  17, 18  #
##################################

def preprocess(filename, data_dir):
    filepath = os.path.join(data_dir, filename)
    f_in = open(filepath, 'r')
    processed_filepath = filepath[:-4] + "_PROED" + filepath[-4:]
    if not os.path.exists(processed_filepath):
        f_out = open(processed_filepath, 'w')
        line = f_in.readline()
        while line:
            line = line.lstrip(string.digits)
            line = line.replace("(e1,e2)", " (e1,e2)")
            line = line.replace("(e2,e1)", " (e2,e1)")
            line = line.lstrip('\t')
            line = line.replace('"', '')
            line = line.replace("<e1>", "<e1> ")
            line = line.replace("</e1>", " </e1>")
            line = line.replace("<e2>", "<e2> ")
            line = line.replace("</e2>", " </e2>")
            line = line.replace(',', ' ,')
            line = line.replace('.', ' .')
            line = line.replace('!', ' !')
            line = line.replace('?', ' ?')
            line = line.replace(':', ' :')
            line = line.replace('$', ' $')
            line = line.replace('\n', ' \n')
            
            if not line[0:7] == "Comment":
                f_out.write(line)
            
            line = f_in.readline()
        f_out.close()
        print(filename, "processed")
    else:
        print(filename, "existed")
    f_in.close()
    
    
def read_data_sets(data_dir, padding=False):
    train_filepath = os.path.join(data_dir, "TRAIN_FILE.txt")
    test_filepath = os.path.join(data_dir, "TEST_FILE.txt")
    preprocess("TRAIN_FILE.txt", data_dir)
    preprocess("TEST_FILE.txt", data_dir)
    if os.path.exists('word2vec.npy') == False:
        print("please train word vectors first")
        return
    else:
        train_filepath = os.path.join(data_dir, "TRAIN_FILE_PROED.txt")
        test_filepath = os.path.join(data_dir, "TEST_FILE_PROED.txt")
        semdata = SemData(train_filepath, test_filepath, padding)
    return semdata

    
class Data(object):
    def __init__(self, filepath, padding):
        max_length = 97
        n_embedding = 300
        self.pos = 0
        self.sentences = []
        self.labels = None
        labels = []
        embeddings = np.load("word2vec.npy").item()
        label_dict = {"Other": 0, "Cause-Effect": 2, "Component-Whole": 4, "Entity-Destination": 6, 
                      "Product-Producer": 8, "Entity-Origin": 10, "Member-Collection": 12,
                      "Message-Topic": 14, "Content-Container": 16, "Instrument-Agency": 18}
        f_data = open(filepath, 'r')
        
        line = f_data.readline()
        while line:
            sentence = []
            line = line.split(' ')
            line.remove('\n')
            sentence = [embeddings[word] for word in line]
            if padding==True:
                zero_length = max_length - len(sentence)
                for i in range(zero_length):
                    sentence.append(np.zeros([n_embedding], float))
            self.sentences.append(np.array(sentence))
            
            line = f_data.readline()
            line = line.split(' ')
            label = label_dict[line[0]]
            if not label == 0 and line[1] == "(e1":
                label -= 1
            labels.append(label)
            
            line = f_data.readline()
            line = f_data.readline()
        self.sentences = np.array(self.sentences)
        self.labels = np.zeros([len(labels), 19])
        for i in range(len(labels)):
            self.labels[i, labels[i]] = 1
        
    def next_batch(self, batch_size):
        if self.pos+batch_size > len(self.sentences) or self.pos+batch_size > len(self.labels):
            self.pos = 0
        res = (self.sentences[self.pos:self.pos+batch_size], self.labels[self.pos:self.pos+batch_size])
        self.pos += batch_size
        return res
    
    
class SemData(object):
    def __init__(self, train_filepath, test_filepath, padding):
        self.train = Data(train_filepath, padding)
        self.test = Data(test_filepath, padding)
    

