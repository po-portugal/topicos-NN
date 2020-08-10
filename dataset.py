import numpy as np
import matplotlib.pyplot as plt
import pdb
import csv
import os


class Dataset():
    def __init__(self, path_to_dir, target, label_to_num, preprocess="single_card"):
        self.dataset_dir = path_to_dir
        self.target = target
        self.label_to_num = label_to_num

        self.labels_file_path = os.path.join(
            self.dataset_dir, preprocess+"_"+self.target+"_labels.csv")
        self.images_folder = os.path.join(self.dataset_dir, target)

        n = len(self.label_to_num)-1
        if n > 0:
            def one_hot_enc(label):
                return [int(i == self.label_to_num[label]) for i in range(n-1, -1, -1)]
            self.enc = one_hot_enc
        else:
            self.enc = lambda label: [label]

        if preprocess == "scaled":
            proc_in, proc_out = self.processors_scaled()
        elif preprocess == "original":
            proc_in, proc_out = self.processors_original()
        elif preprocess == "single_card":
            proc_in, proc_out = self.processors_single_card()
        elif preprocess == "yolo_full":
            proc_in, proc_out = self.processors_yolo_full()
        elif preprocess == "yolo_pos":
            proc_in, proc_out = self.processors_yolo_pos()
        else:
            raise ValueError

        self.X_meta, self.Y, self.header = getInputs(
            self.labels_file_path, proc_in, proc_out, ',')


    def get_images(self):
        path = os.path.join(self.dataset_dir, "images_norm", self.target)
        self.X = np.array([load_norm_img(path, x_meta[0])
                           for x_meta in self.X_meta])

        #shape = list(self.X.shape)+[1]
        #self.X = self.X.reshape(shape)

    def save_scaled_version(self):
        X_scaled, Y_scaled = scale_boundaries(self.X_meta, self.Y)
        header_scaled = [self.header[0]]+self.header[3:]
        saveInputs(self.dataset_dir+"/scaled_"+self.target+"_labels.csv",
                   X_scaled, Y_scaled, header_scaled, ',')

    def save_yolo_pos_version(self, grid_size_x, grid_size_y):
        X, Y = self.get_yolo_pos_labels(grid_size_x, grid_size_y)
        header = ["To implement :("]
        saveInputs(self.dataset_dir+"/yolo_pos_"+self.target +
                   "_labels.csv", X, Y, header, ',')

    def save_yolo_full_version(self, grid_size_x, grid_size_y):
        X, Y = self.get_yolo_full_labels(grid_size_x, grid_size_y)
        header = ["To implement :("]
        saveInputs(self.dataset_dir+"/yolo_full_" +
                   self.target+"_labels.csv", X, Y, header, ',')

    def get_yolo_pos_labels(self, grid_size_x, grid_size_y):
        step_x = 1.0/grid_size_x
        step_y = 1.0/grid_size_y

        names = [x[0] for x in self.X_meta]
        Y_yolo, X_yolo = [], []
        Y_array = np.array(self.Y, dtype=object)
        X_array = np.array(names, dtype=str)
        X_set = set(names)
        for name in X_set:
            img_labels = Y_array[X_array == name]
            label_yolo = np.zeros((grid_size_x, grid_size_y)).tolist()
            for label in img_labels:
                xc, yc = (label[-4:-2]+label[-2:])/2
                grid_x_index = int(xc/step_x)
                grid_y_index = int(yc/step_y)
                label_yolo[grid_y_index][grid_x_index] = label[0]
            Y_yolo.append(np.array(label_yolo).flatten().tolist())
            X_yolo.append([name])

        return X_yolo, Y_yolo

    def get_yolo_full_labels(self, grid_size_x, grid_size_y):
        step_x = 1.0/grid_size_x
        step_y = 1.0/grid_size_y
        centers = [(int(step_x/2+i*step_x), int(step_y/2+j*step_y))
                   for i in range(grid_size_x-1) for j in range(grid_size_y-1)]
        upper_corner = np.array([[[i*step_x, j*step_y]
                                  for i in range(grid_size_x)] for j in range(grid_size_y)])

        names = [x[0] for x in self.X_meta]
        Y_yolo, X_yolo = [], []
        Y_array = np.array(self.Y, dtype=object)
        X_array = np.array(names, dtype=str)
        X_set = set(names)
        for name in X_set:
            img_labels = Y_array[X_array == name]
            label_yolo = np.zeros(
                (grid_size_x, grid_size_y, len(self.Y[0]))).tolist()
            for label in img_labels:
                img_center = (label[-4:-2]+label[-2:])/2
                h, w = (label[-2:]-label[-4:-2])
                grid_x_index = int(img_center[0]/step_x)
                grid_y_index = int(img_center[1]/step_y)
                #label_yolo[grid_y_index,grid_x_index] = 1
                center_dist = (
                    img_center-upper_corner[grid_y_index][grid_x_index])
                label_scaled = np.concatenate(
                    (label[:-4], [center_dist[0]/step_x, center_dist[1]/step_y, w/step_x, h/step_y])).flatten().tolist()
                label_yolo[grid_y_index][grid_x_index] = label_scaled
            Y_yolo.append(np.array(label_yolo).flatten().tolist())
            X_yolo.append([name])

        return X_yolo, Y_yolo

    def processors_original(self):
        def process_in(row): return [row[0]]+ [int(r) for r in row[1:3]]
        def process_out(row): 
            return self.enc(row[3])+[int(r) for r in row[4:]]
        return process_in, process_out

    def processors_scaled(self):
        def process_in(row): return [row[0]]
        def process_out(row): return self.enc(row[1])+[float(r) for r in row[2:]]
        return process_in, process_out

    def processors_single_card(self):
        return self.processors_scaled()

    def processors_yolo_pos(self):
        def process_in(row): return [row[0]]
        def process_out(row): return [self.enc(r) for r in row[1:]]
        return process_in, process_out

    def processors_yolo_full(self):
        def process_in(row): return [row[0]]

        def process_out(row):
            n = int((len(row)-1)/5)
            cells = [self.enc(row[i])+[float(r) for r in row[i+1:i+5]]
                     for i in range(1, n*5, 5)]
            return [i for l in cells for i in l]
        return process_in, process_out

    def analiseSamples(self):
        """ Print Histogram for the output varable"""
        n, bins, patches = plt.hist(
            x=self.Y, bins=20, density=False, align='left')

        #plt.title("Hitograma do número de bicicletas")
        #plt.axis([0, 10, 0, 1])
        plt.grid(True)
        plt.ylabel("frequência")
        plt.xlabel("faixa do atributo")
        plt.show()

    def get_input_output(self):
        return self.X, self.Y

    def set_labels_slice(self,model_name):
        self.Y = np.array(self.Y)
        if model_name == "yolo":
          #self.Y = [ self.Y[:,:6], self.Y[:,6:] ]
          self.Y = [ self.Y[:,6:] ]

        elif model_name == "classifier":
          self.Y = self.Y[:,:6]
        else :
          raise ValueError("model_name not valid : '",model_name,"'")

    def save_single_card_dataset(self):
        names = [x[0] for x in self.X_meta]
        nums = [names.count(name) for name in names]
        X_save = [x for x, num in zip(self.X_meta, nums) if num == 1]
        Y_save = [y for y, num in zip(self.Y, nums) if num == 1]
        path_to_new_dataset = os.path.join(
            self.dataset_dir, "single_card_"+self.target+"_labels.csv")
        saveInputs(path_to_new_dataset, X_save, Y_save, self.header, ',')

    def print_check(self,model_name):
        index = np.random.randint(len(self.X_meta))
        print("Print check for ",self.target)
        print("Input file ", self.X_meta[index])
        print("Input image ", self.X[index])
        
        if model_name == "yolo":
          print("Label ", self.Y[0][index], self.Y[1][index])
        elif model_name == "classifier":
          print("Label ", self.Y[index])
        else:
          raise ValueError("model_name not found: '",model_name,"'")


def scale_boundaries(X, Y):
    X_scaled, Y_scaled = [], []
    for x, y in zip(X, Y):
        X_scaled.append([x[0]])
        y[1] = float(y[1]/x[1])
        y[3] = float(y[3]/x[1])
        y[2] = float(y[2]/x[2])
        y[4] = float(y[4]/x[2])
        Y_scaled.append(y)
    return X_scaled, Y_scaled


def getInputs(path, preprocess_in, preprocess_out, delim):
    X, Y = [], []
    with open(path) as fd:
        csvFd = csv.reader(fd, delimiter=delim)
        header = next(csvFd)
        for row in csvFd:
            X.append(preprocess_in(row))
            Y.append(preprocess_out(row))
    return X, Y, header


def saveInputs(path, X, Y, header, delim):
    with open(path, 'w') as fd:
        csvFd = csv.writer(fd, delimiter=delim)
        csvFd.writerow(header)
        for x, y in zip(X, Y):
            csvFd.writerow(x+y)
    return

def load_norm_img(folder, filename):
    base_name, _ = filename.split('.')
    path_to_file = os.path.join(folder, base_name+".npy")
    return np.load(path_to_file)
