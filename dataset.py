from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pdb
import csv
import os

class Dataset():
    def __init__(self,path_to_dir,target):
        labels_file   = os.path.join(path_to_dir,target+"_labels.csv")
        images_folder = os.path.join(path_to_dir,target)

        proc_in, proc_out = self.processors(images_folder)

        self.X, self.Y,_ = getInputs(labels_file,proc_in,proc_out,',')

    def processors(self,target_dir):
        #process_in  = lambda row:[row[0],[int(r) for r in row[1:3]]]
        #process_in  = lambda row:[load_norm_img(target_dir,row[0]),[int(r) for r in row[1:3]]]
        process_in  = lambda row:load_norm_img(target_dir,row[0])
        label_to_num = {
                "king":0,
                "queen":1,
                "jack":2,
                "nine":3,
                "ten":4,
                "ace":5,
        }
        n = len(label_to_num)
        enc = lambda label:[int(i==label_to_num[label]) for i in range(n,-1,-1)]
        process_out = lambda row:enc(row[3])+[int(r) for r in row[4:]]
        
        return process_in,process_out
    
    def analiseSamples(self):
        """ Print Histogram for the output varable"""
        n,bins,patches = plt.hist(x=self.Y,bins=20,density=False,align='left')
    
        plt.title("Hitograma do número de bicicletas")
        #plt.axis([0, 10, 0, 1])
        plt.grid(True)
        plt.ylabel("frequência")
        plt.xlabel("faixa do atributo")
        plt.show()

    def get_input_output(self):
        return self.X, self.Y


def getInputs(path,preprocess_in,preprocess_out,delim):
    X,Y = [],[]
    with open(path) as fd :
        csvFd = csv.reader(fd,delimiter=delim)
        header = next(csvFd)
        for row in csvFd :
            X.append(preprocess_in(row))
            Y.append(preprocess_out(row))
    return X,Y,header

def load_norm_img(folder,filename):
    base_name, _ = filename.split('.')
    path_to_file = os.path.join(folder,base_name+".npy")
    return np.load(path_to_file)
