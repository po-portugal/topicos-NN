from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pdb
import csv
import os

class Dataset():
    def __init__(self,path_to_dir,target,preprocess="basic"):
        self.labels_file   = os.path.join(path_to_dir,target+"_labels.csv")
        self.images_folder = os.path.join(path_to_dir,target)

        if preprocess=="basic":
            proc_in, proc_out = self.processors_basic(self.images_folder)
        elif preprocess=="basic_original":
            proc_in, proc_out = self.processors_basic_original(self.images_folder)
        elif preprocess=="complete":
            proc_in, proc_out = self.processors_complete(self.images_folder)
        elif preprocess=="complete_original":
            proc_in, proc_out = self.processors_complete_original(self.images_folder)
        else:
            raise ValueError

        self.X, self.Y,header = getInputs(self.labels_file,proc_in,proc_out,',')

    def save_scaled_version(self):
        X_scaled, Y_scaled = scale_boundaries(self.X,self.Y)
        header_scaled = [header[0]]+header[3:]
        saveInputs("images_norm/"+target+"_labels_scaled.csv",X_scaled,Y_scaled,header_scaled,',')

    def processors_basic_original(self,target_dir):
        process_in_basic  = lambda row:[row[0],*[int(r) for r in row[1:3]]]
        process_out_basic = lambda row:[row[3]]+[int(r) for r in row[4:]]
        return process_in_basic,process_out_basic

    def processors_complete_original(self,target_dir):
        process_in_mat  = lambda row:load_norm_img(target_dir,row[0])
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
        process_out_one_hot = lambda row:enc(row[3])+[int(r) for r in row[4:]]
        
        return process_in_mat,process_out_one_hot
    
    def processors_basic(self,target_dir):
        process_in_basic  = lambda row:[row[0]]
        process_out_basic = lambda row:[row[1]]+[int(r) for r in row[2:]]
        return process_in_basic,process_out_basic

    def processors_complete(self,target_dir):
        process_in_mat  = lambda row:load_norm_img(target_dir,row[0])
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
        process_out_one_hot = lambda row:enc(row[1])+[int(r) for r in row[2:]]
        
        return process_in_mat,process_out_one_hot

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


def scale_boundaries(X,Y):
    X_scaled, Y_scaled = [], []
    for x,y in zip(X,Y):
        X_scaled.append([x[0]])
        if x[1]==960 and x[2]==540 :
            y[1] = _scale_x(y[1])
            y[3] = _scale_x(y[3])
            y[2] = _scale_y(y[2])
            y[4] = _scale_y(y[4])
        Y_scaled.append(y)
    return X_scaled, Y_scaled

def getInputs(path,preprocess_in,preprocess_out,delim):
    X,Y = [],[]
    with open(path) as fd :
        csvFd = csv.reader(fd,delimiter=delim)
        header = next(csvFd)
        for row in csvFd :
            X.append(preprocess_in(row))
            Y.append(preprocess_out(row))
    return X,Y,header

def saveInputs(path,X,Y,header,delim):
    with open(path,'w') as fd :
        csvFd = csv.writer(fd,delimiter=delim)
        csvFd.writerow(header)
        for x,y in zip(X,Y) :
            csvFd.writerow(x+y)
    return

def load_norm_img(folder,filename):
    base_name, _ = filename.split('.')
    path_to_file = os.path.join(folder,base_name+".npy")
    return np.load(path_to_file)

def _scale_x(original):
    return _scale(original,960,378)

def _scale_y(original):
    return _scale(original,540,504)

def _scale(original,original_dim,scaled_dim):
    scaling_factor = scaled_dim/original_dim
    rescaled = original*scaling_factor
    return int(rescaled)

