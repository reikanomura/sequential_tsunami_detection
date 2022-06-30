"""
Performs Proper Orthogonal Decomposition on input
"""
import os
import logging
import numpy as np
import dask.array as da
import beautyfun as bf
import graphing as gf
lg = logging.getLogger('my_logger')

def da_svd_wrap(mat):
    if mat.shape[0] < mat.shape[1]:
        mat = mat.T
        transposed = True
    else:
        transposed = False

    mat = mat.rechunk({0:'auto', 1:-1})

    U, s, VT = da.linalg.tsqr(mat, compute_svd=True)
    U, s, VT = da.compute(U, s, VT)

    if transposed:
        [U, VT] = [VT.T, U.T]

    return U, s, VT

def solve(loadfiles, dir_npy, dir_graphs):
    lg.debug('Reading data')
    Xmat = bf.readdata(loadfiles)

    lg.debug('Performing SVD')
    U, s, VT = da_svd_wrap(Xmat)
    gf.make_scree(s, 90.5, bf.makedirs(dir_graphs + 'explained_variance.png'))
    
    alpha = U * s[..., None, :] 
    lg.debug('Alpha_shape :' + str(alpha.shape))
    lg.debug('Mode shape :' + str(VT.shape))

    bf.saveda(dir_npy, 'U', U)
    bf.saveda(dir_npy, 's', s)
    bf.saveda(dir_npy, 'VT', VT)
    bf.saveda(dir_npy, 'alpha', alpha)


    return U, s, VT

def check(ncase, npnts, nrecord, cases_dir, ncheck_pnts, list_eval, U, s, VT, dir_graphs):
    ncheck_case = np.random.random_integers(ncase) - 1

    lg.debug(cases_dir[ncheck_case])
    lg.debug(list_eval[ncheck_pnts])

    testfile = bf.mk_filepath([cases_dir[ncheck_case]], list_eval)
    Xmat_test = bf.readdata(testfile)
    Xmat_test = Xmat_test.compute()
    lg.debug(str(Xmat_test.shape))

    A = U[:,:len(list_eval)] * s[..., None, :len(list_eval)]
    lg.debug('Alpha shape: '+ str(A.shape))

    for imode in range(10, 50, 5):
        recon = A[:, :imode] @ VT[:imode, :]
        lg.debug('recon shape: ' + str(recon.shape))

        rmse =gf.make_wave(nrecord, Xmat_test[:,ncheck_pnts],
                           recon[nrecord*(ncheck_case):nrecord*(ncheck_case+1),ncheck_pnts],
                           ncase, imode, 'POD Wave Reconstruction (# of modes:{})'.format(imode), 
                           dir_graphs + 'pod_reconstruct_{:03d}.png'.format(imode))
        lg.debug('recon RMSE at r={}: '.format(str(imode)) + str(rmse))

