from scipy.stats import norm
import numpy as np
import beautyfun as bf
import logging
import graphing as gf
import os
from multiprocessing import Process
lg = logging.getLogger('my_logger')

def observe(A_dash, observations, n_dim, time, dir_res):
    observations = A_dash[time, :n_dim]
    return observations

def predict(nrecord, A, particles, n_dim, time, dir_res):
    for i in range(n_dim):
        particles[:, i] = A[time::nrecord, i]
    return particles

def update(particles, nparticle, n_dim, weights, observations,sigma):
    for i in range(n_dim):
        weights *= norm(particles[:, i], sigma[i]).pdf(observations[i])

    weights = np.where(weights >= 0.9, 0.9, weights)
    weights += 1.e-50      # avoid round-off to zero
    weights /= sum(weights) # normalize
    return weights

def xlim_ylim(A, n_dim, dir_res):
    xlim = (np.amin(A[:,0]),np.amax(A[:,0]))
    ylim = (np.amin(A[:,1]),np.amax(A[:,1]))
    return xlim, ylim

def fun(dt, nrecord, test_cases, train_cases, U, s, VT, nparticle, n_dim, iout, niters, ncheck_pnt,dir_res, vald_txt):
    
    sigma = 0.1 * np.sqrt(s)
    A = U[:,:n_dim] * s[..., None, :n_dim]
    A_dash = np.load(os.path.join(dir_res,'alpha_dash.npy'))
    lg.debug('Size of alpha: {}'.format(A.shape))
    lg.debug('Size of VT: {}'.format(np.shape(VT[:n_dim, :])))

    similar_cases =list() 

    procs = []
    for i, case in enumerate(test_cases):
        # initialize
        idata = nrecord * i
        observations = np.empty(n_dim)
        particles = np.empty((nparticle, n_dim))
        weights = np.ones(nparticle) / nparticle

        #xlim, ylim = xlim_ylim(A, n_dim, dir_res)
        xlim = (-13.5, 13.5)
        ylim = (-6.0, 6.0)

        wei_all = np.empty((niters,nparticle))

        for istep in range(niters):
            mytime = int((istep+1)* dt * iout)
            observations = observe(A_dash, observations, n_dim, idata, dir_res)
            particles = predict(nrecord, A, particles, n_dim, idata%nrecord, dir_res)
            lg.debug('time step {0:d} [sec]'.format(mytime))
            weights = update(particles, nparticle, n_dim, weights, observations, sigma)
            p = Process(target=gf.make_pf, args=(istep, mytime, particles, weights, observations, xlim, ylim,
              bf.makedirs(dir_res + 'figures/pf{:02d}/pf_{:04d}.png'.format(i, istep))))
            procs.append(p)
            p.start()
            idata += iout 

            wei_all[istep,:] = weights

        for p in procs:
            p.join()

        procs = []

        path = os.path.join(dir_res, 'weights-{}'.format(case))
        np.save(path, wei_all)

        mcase =np.where(weights == np.max(weights))[0][0] 
        most_case = train_cases[mcase]


        tsta, tend  = nrecord * mcase, nrecord * (mcase + 1)
        dsta, dend  = nrecord * i,     nrecord * (i+1)

        gf.alpha_comparison(i, case, mcase, most_case, 
                            tsta, tend, dsta, dend, 
                            A, A_dash, VT, n_dim, ncheck_pnt, 
                            dt, nrecord, mytime, dir_res)

        lg.debug('Most similar scenario: {}'.format(most_case))
        lg.debug('Tested scenario: {}'.format(os.path.basename(case)))

        similar_cases.append(most_case)

    np.savetxt(os.path.join(dir_res, vald_txt), similar_cases, fmt='%s')
