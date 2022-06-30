import logging
from random import shuffle
import itertools 
import numpy as np
import beautyfun as bf
import graphing as gf
lg = logging.getLogger('my_logger')



def fun(nrecord, ncases, npnts, ncheck_pnt, U, s, VT, eventfiles, list_cases, list_pnts, dir_res):

    modes = VT[:npnts, :]

    X_test = bf.readdata(eventfiles)
    X_test = X_test.compute()

    lg.debug('modes shape: ' + str(modes.shape))
    lg.debug('X_test shape: ' + str(X_test.shape))

    A_dash = np.matmul(X_test, np.linalg.pinv(modes))

    lg.debug('A_dash shape: ' + str(A_dash.shape))
    bf.saveda(dir_res, 'alpha_dash', A_dash)
    bf.saveda(dir_res, 'X_dash', X_test)

    lg.debug(list_pnts[ncheck_pnt])

    for nmodes in range(10, 50, 5):
      for index, icase in enumerate(list_cases[0:2]):
            recon = A_dash[:, :nmodes] @ VT[:nmodes, :]
            lg.debug('recon shape: ' + str(recon.shape))
        
            xtmp = X_test[nrecord*(index):nrecord*(index+1), :]
            rtmp = recon[nrecord*(index):nrecord*(index+1), :]
            lg.debug('xtmp shape: ' + str(xtmp.shape))
            lg.debug('rtmp shape: ' + str(rtmp.shape))
            rmse = gf.make_wave(nrecord, xtmp[:, ncheck_pnt], rtmp[:, ncheck_pnt],
                                ncases, nmodes, 'Psudo Inverse Reconstruction (# of modes:{})'.format(nmodes),
                                dir_res + 'figures/psudo_inv_{}_{:03d}.png'.format(icase, nmodes))

            lg.debug('psudo RMSE at r={}'.format(str(nmodes)) + str(rmse))
