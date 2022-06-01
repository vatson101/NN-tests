import matplotlib
import numpy as np
import ast
import random
import matplotlib.pyplot as plt
import matplotlib.image as matimg
import time
import sys

def recursion(a):
    print (a)
    a = a + 1
    if a>16:
        return a
    else:
        return recursion(a)
def f(x):
    return 1/(1 + np.exp(-x))

def f_der(x):
    return f(x)*(1-f(x))

def matrix_f_der(mat):
    out_arr = np.zeros(len(mat), len(mat[0]))
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            out_arr[i, j] = f_der(mat[i, j])

def nn_step(w, in_x:np.ndarray, b):
    print(w,'w\n', in_x, 'in\n', b, 'b')
    h = w.dot(np.vstack(in_x))
    print(h,'h')
 
    h = h + np.vstack(b)
    print(h)
    for i in range(0, len(h)):
        h[i][0] = round(f(h[i][0]), 3)
    print(h, 'f(h)\n')
    
    return h



def interpolate(x):
    return x/255

def intpol_list(list):
    for i in range(len(list)):
        list[i] = list[i]/255
    
    return list

def error_funk(x, x_correct): #x and x_correct are vectors
    if len(x) != len(x_correct): return None

    sum = 0
    for z, z_ in zip(x, x_correct):
        sum += (z - z_)**2
    return sum/2

def num_to_vector(num): #to get a neural representation of a digit
    vector = np.zeros((10))
    vector[num] = 1
    return vector

def random_nn_gen(neurons_count):
    layers = []
    for i in range(0, len(neurons_count) - 1):
        arr = np.zeros((neurons_count[i+1], neurons_count[i]))
        for g in range(len(arr)):
            for j in range(len(arr[g])):
                arr[g, j] = round(random.gauss(0, 2.5), 2)
        
        vect = []
        for s in range(0, neurons_count[i + 1]):
            vect.append(round(random.gauss(0, 2.5), 2))
        
        vect = np.array(vect)
        layers.append(Layer(arr, vect))
    
    return layers

def draw_bytes(bytearray):
    picc = []
    for i in range(28):
        arr = []
        for j in range(28):
            arr.append(bytearray.pop(0))
        picc.append(arr)
    pic = np.array(picc)
    plt.imshow(pic, cmap='gray')
    plt.show()
    

def open_images():
    file = open("train-images.idx3-ubyte", "rb")
    print(int.from_bytes(file.read(4), 'big'))
    print(int.from_bytes(file.read(4), 'big'))
    print(int.from_bytes(file.read(4), 'big'))
    print(int.from_bytes(file.read(4), 'big'))
    return file

def open_labels():
    file = open("train-labels.idx1-ubyte", "rb")
    print(int.from_bytes(file.read(4), 'big'))
    print(int.from_bytes(file.read(4), 'big'))
    return file

class Layer:
    def __init__(self, w, b) -> None:
        self.w = w
        self.b = b
def something(layer:Layer, h):
    pass
class Neural_Network:
    def __init__(self, layers) -> None:
        self.layers = layers

    def compute(self, x):
        h = x
        for layer in self.layers:
            h = nn_step(layer.w, h, layer.b)
        
        return h

    def file_to_layers(self, filename):
        lines = tuple(open(filename, 'r'))
        layers = []
        temp = [] #dimentions or layer
        for line in lines:
            temp.append(line)
            if len(temp) == 2:
                try:
                    layers.append(Layer(np.array(ast.literal_eval(temp[0])), 
                                        np.array(ast.literal_eval(temp[1]))))
                except:
                    return None
                temp = []
        self.layers = layers
        return layers
    
    def write_layers(self, filename):
        f = open(filename, "a")
        for layer in self.layers:
            f.write(str(layer.w.tolist()))
            f.write('\n')
            f.write(str(layer.b.tolist()))
            f.write('\n')

        f.close()
    
    def compute(self, x_array):
        h = x_array
        for layer in self.layers:
            h = nn_step(layer.w, h, layer.b)
        return h

b = recursion(1)
print(b)
# random.seed(10)
# print(np.zeros((2,3)))
# nn = Neural_Network(random_nn_gen([2, 3, 4]))
# nn.compute(np.array([3,4]))
# nn = Neural_Network(random_nn_gen([784, 16, 16, 10]))



# file1 = open_images()
# file2 = open_labels()

# pic_bytes = list(file1.read(784))

# image = intpol_list(pic_bytes)
# answer = nn.compute(image)

# print('result: ' + str(answer))
# print(error_funk(nn.compute(image), num_to_vector(5)))

# pic = np.array(image)

# print(int.from_bytes(file2.read(1), byteorder=sys.byteorder))
# draw_bytes(pic_bytes)


# file2.close()
# file1.close()
