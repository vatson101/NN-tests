import numpy as np
import ast

def write_array(arr, filename):
    f = open(filename, "a")
    f.write(str(arr.tolist()))
    f.write('\n')
    f.close()

# def read_array(filename):

a = np.array([
     [1],
     [-1],
     [3]
 ])


write_array(a, "Neural 1.txt")
