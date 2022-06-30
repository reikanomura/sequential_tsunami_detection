"""
This module contains functions which are shared
across modules to make them look more pretty.
"""
import os
import logging
import numpy as np
import dask.array as da
from multiprocessing import Pool
lg = logging.getLogger('my_logger')

def makedirs(path):
    """ Create directories if they do not exist
       and return the path """
    dirs = path.rsplit('/', 1)[0]
    try:
        os.makedirs(dirs)
    except OSError:
        if not os.path.isdir(dirs):
            raise
    return path


def save_lst(lst, path):
    np.savetxt(path, lst, fmt='%s', delimiter='\n')

def load_lst(txtfile):
    mylist = np.loadtxt(txtfile, dtype='unicode').tolist()
    return mylist


def combine_2lists(cases_dir, eval_pnts):
    all_dir = []
    for case_dir in cases_dir:
        all_dir.append([case_dir + '/' + s for s in eval_pnts])
    return all_dir

def mk_filepath(cases_dir, eval_pnts):
    all_dir = []
    for case_dir in cases_dir:
        all_dir.append([case_dir + '/' + s for s in eval_pnts])
    return all_dir


def readdata_i(fds):
    if isinstance(fds, str):
        fds = [fds]
    x = np.array([])
    for fd in fds:
        f = str(fd)
        t = np.genfromtxt(f, delimiter=',',usecols=[1])
        x = np.append(x, t) # x is a long vector with t stuck on the end
    x = np.reshape(x, (-1, t.shape[0])).T # Chop up x to create a 2d array
    #dax = da.from_array(x)
    #lg.debug(dax)
    return x

def divide_chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        res.append(l[i:i + n])
    return res

def readdata(all_dir):
    nchunks = divide_chunks(all_dir, 30)
    init = True
    for ichunk in nchunks:
        p = Pool()
        temp = p.map(readdata_i, ichunk)
        temp = np.concatenate(temp, axis=0)
        
        if init:
            dax = da.from_array(temp, chunks="auto")
            #dax = temp.rechunk((temp.shape[0], temp.shape[1]))
            init = False
        else:
            temp = da.from_array(temp, chunks="auto")
            #temp = temp.rechunk((temp.shape[0], temp.shape[1]))
            dax  = da.concatenate([dax, temp], axis=0)
        lg.debug(dax)
    return dax

def saveda(fd, fn, X):
    makedirs(fd)
    lg.debug(fn + ' ' + str(X.shape))
    try:
        da.to_npy_stack(fd + fn + '/', X)
        lg.debug('Saved ' + fn + ' as np stack')
    except:
        np.save(fd + fn + '.npy', X)
        lg.debug('Saved ' + fn + ' as np')
