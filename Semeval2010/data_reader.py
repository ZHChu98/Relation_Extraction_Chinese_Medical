import tensorflow as tf
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
    
    
def read_data_sets(data_dir):
    train_filepath = os.path.join(data_dir, "TRAIN_FILE.txt")
    test_filepath = os.path.join(data_dir, "TEST_FILE.txt")
    preprocess("TRAIN_FILE.txt", data_dir)
    preprocess("TEST_FILE.txt", data_dir)
    
    
    
    

