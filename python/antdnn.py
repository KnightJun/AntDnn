# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:45:55 2017

@author: KnightJun
"""

import numpy as np

def np_to_file(npdata, filename):
    if npdata.dtype != np.float32:
        raise "only support float32 dtype"
    with open(filename, 'wb') as f:
        f.write("ant".encode())  # file flag
        f.write(b'\0')
        __np_to_f(npdata, f)
    return True

def __np_to_f(npdata, f):
    f.write('tsd'.encode()) # data type
    f.write(b'\0')
    header = [len(npdata.shape)]
    header += list(npdata.shape)
    f.write(np.array(header).tobytes())
    f.write(npdata.tobytes())
    
def nplist_to_file(nplist, filename):
    with open(filename, 'wb') as f:
        f.write("ant".encode())  # file flag
        f.write(b'\0')
        f.write('lis'.encode()) # data type
        f.write(b'\0')
        header = [len(nplist)]
        f.write(np.array(header).tobytes())
        for npdata in nplist:
            __np_to_f(npdata, f)
            

if __name__ == "__main__":
    tdata = np.ones((3, 3, 4), dtype=np.float32)
    np_to_file(tdata, "test.antts")