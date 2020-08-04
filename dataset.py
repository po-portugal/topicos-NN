from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self,X_raw,Y_raw,header=None):
        self.X, self.Y = X_raw, Y_raw
        self.header = header
        if header is not None:
            self.X_names = self.header[:-1]
            self.Y_names = self.header[-1:]

    def split(self,test_size,val_size,seed):
        self.test_size = test_size
        self.val_size = val_size
        self.train_size_prct = 100*(1.0-val_size-test_size)

        X_final_train, self.X_test,Y_final_train,self.Y_test = \
        train_test_split(self.X,self.Y,test_size=test_size)
    
        self.X_train,self.X_val,self.Y_train,self.Y_val = \
        train_test_split(
            X_final_train,
            Y_final_train,
            test_size=val_size/(1-test_size),
            random_state=seed)

        self.test_size  = len(self.X_test)
        self.train_size = len(self.X_train)
        self.input_size = len(self.X[0])
        self.output_size = 1#len(Y[0])

    def normalise(self):
        self.X_raw, self.Y_raw = self.X, self.Y
        self.X, self.Y = self.normaliseX(self.X_raw), self.normaliseY(self.Y_raw)

    def normaliseX(self,X):
        """Normalizar os parâmetros de entrada"""
    
        atts = self.getAttributes(X)
        att_normal = []
        for att in atts:
            array = np.array(att)
            upp,low = np.amax(array),np.amin(array)
            band = upp - low
            att_normal.append((array - low)/band)
            #import pdb; pdb.set_trace()
        
        return np.array(self.packAttributes(att_normal))

    def normaliseY(self,Y):
        """Normalizar os parâmetros de entrada"""
        upp,low = np.amax(Y),np.amin(Y)
        band = upp - low
        return (Y - low)/band
    
    def getAttributes(self,X):
        """Separate Input samples into the respective attributes"""
        return [list(i) for i in zip(*X)]
    
    def packAttributes(self,atts):
        """Undoes getAttributes"""
        return [list(i) for i in zip(*atts)]

    def analiseFeatures(self):
        atts = np.array(self.getAttributes(self.X))
        ind = list(range(0,len(atts)))
        means = atts.mean(1)
        stds  = atts.std(1)
        
        print("\\toprule")
        print("\\toprule")
        print("Índice & Atributo & Média & Desvio padrão \\\\")
        print("\\midrule")
        for i,n,m,s in zip(ind,self.X_names,means,stds):
            print("%s & %s & %.2f & %.3f \\\\"%(i,n,m,s))
        print("\\bottomrule")
        print("\\bottomrule")
        return
    
    def analiseSamples(self):
        """ Print Histogram for the output varable"""
        n,bins,patches = plt.hist(x=self.Y,bins=20,density=False,align='left')
    
        plt.title("Hitograma do número de bicicletas")
        #plt.axis([0, 10, 0, 1])
        plt.grid(True)
        plt.ylabel("frequência")
        plt.xlabel("faixa do atributo")
        plt.show()
