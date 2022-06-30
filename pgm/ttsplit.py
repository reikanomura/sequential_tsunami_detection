"""
This module takes a list and splits it for cross validation.
Will implement K-means clustering for selecting test cases
in a future version
"""
from random import shuffle

def fun(lst_all, nlarn, ntest):
    """The main function of the ttsplit module"""
    shuffle(lst_all)
    nall = len(lst_all)

    if nall == nlarn:
      lst_larn = lst_all

    elif nall != nlarn:
      lst_larn = lst_all[0:nlarn]

    lst_test = lst_all[(nall-ntest):]
    return lst_test, lst_larn
