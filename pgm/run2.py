"""
Editor & Maintainer R. Nomura, J. M. Galbreath
Email: mojotiger608.jg@gmail.com
Email: nomura@irides.tohoku.ac.jp
"""
import time
import logging as lg
import numpy as np
import dask
import dask.array as da
import beautyfun as bf
import ttsplit
import POD
import psudo_inv
import bayesian_update as bu
import os

# Delta t [sec] used in tsunami simulation codes
delt = 5.0  

batch = "TS-2"

inp_dir = '../data/'
inp_scenario = os.path.join(inp_dir, 'cases.txt')
inp_obspnts = os.path.join(inp_dir,'obs_pnts.txt')


txt_test ='list_test'
txt_larn ='list_larn'
txt_vald ='list_pred'

res_dir = bf.makedirs('../res/{}/'.format(batch))
fig_dir = bf.makedirs('../res/{}/figures/'.format(batch))

resta_dir = '../res/{}/'.format(batch)

recon_pnt = 'pnt5_05.asc'

#Bayesian update infer start/end time
init_time =   0  #[sec]
iend_time = 420  #[sec]

#Bayesian update infer step factor (dt * istep) [s]
istep = 1

def main(my_lg):
    """
    This is the main component of the tsunami prediction framework.
    The variables and dimensions of each variable are defined below.
    list_gauges: list with the name for each off-shore evaluation point
    list_allcase: list with the names for all prepared scenarios
    list_test: list with the names for each scenario selected for POD verification test
    list_learn: list with the names for each scenario selected for POD mode extraction
    """

    # Choose which modules to run
    bul_tts = False  # Learn/Test case spliting
    #bul_pod = False  # POD 

    #bul_tts = True  # Learn/Test case spliting
    bul_pod = True  # POD 

    # Read Information of gauges (ovserbation points)
    list_gauges = bf.load_lst(inp_obspnts)
    cpnt = list_gauges.index(recon_pnt)

    ngauges = len(list_gauges)
    my_lg.debug(' The number of gauge for loading:'+ str(ngauges))

    # Spliting into Learn and Test case
    if bul_tts:
        my_lg.debug('### Run Learn/Test Split ###')
        list_allcase = bf.load_lst(inp_scenario)

        list_test, list_larn = ttsplit.fun(list_allcase, nlarn=664, ntest=2)

        bf.save_lst(list_test, res_dir + txt_test )
        bf.save_lst(list_larn, res_dir + txt_larn )

    else:

        my_lg.debug('### Read Previsou Learn/Test Split Results ###')
        list_test = bf.load_lst(resta_dir + txt_test )
        list_larn = bf.load_lst(resta_dir + txt_larn)

    nlarn = len(list_larn)
    ntest = len(list_test)

    testfiles = bf.mk_filepath([(inp_dir + x) for x in list_test], list_gauges)
    larnfiles = bf.mk_filepath([(inp_dir + x) for x in list_larn], list_gauges)

    ndata = len(np.genfromtxt(larnfiles[0][0], delimiter=','))
    my_lg.debug(' Time history data per gauge :'+ str(ndata))

    # POD
    if bul_pod:
        my_lg.debug('### Calculate POD ###')

        Lmat, lamda, Rmat = POD.solve(loadfiles = larnfiles,
                             dir_npy=res_dir, dir_graphs=fig_dir)

        POD.check(ncase=nlarn, npnts=ngauges, nrecord=ndata, 
                  cases_dir=[(inp_dir + x) for x in list_larn],
                  ncheck_pnts = cpnt, list_eval=list_gauges, 
                  U=Lmat, s=lamda, VT=Rmat,
                  dir_graphs=fig_dir)

        dir_res = res_dir
    else:
        my_lg.debug('### Skipping POD ###')
        Lmat  = np.load(os.path.join(resta_dir, 'U.npy'))
        lamda = np.load(os.path.join(resta_dir, 's.npy'))
        Rmat  = np.load(os.path.join(resta_dir, 'VT.npy'))

    my_lg.debug('##############################################')
    my_lg.debug('### ! Real Time Tsunami Prediction start ! ###')
    my_lg.debug('##############################################')

    # `test cases' are treated as on-site data as if they are sampled at real tsunami event hereinafter

    # Psudo_Invi: Identifiyng alpha values of real time event
    my_lg.debug('### Psudo_Inv ###')
    psudo_inv.fun(nrecord=ndata, ncases=nlarn, npnts=ngauges, ncheck_pnt=cpnt,
                  U=Lmat, s=lamda, VT=Rmat, 
                  eventfiles=testfiles,
                  list_cases=list_test,
                  list_pnts=list_gauges,
                  dir_res=res_dir)
    
    # Particle_Filter
    my_lg.debug('### Particle_Filter ###')
    nstep =int(((iend_time-init_time)/delt)/istep)
    iout = istep * delt

    bu.fun(dt=delt,
           nrecord=ndata, test_cases=list_test, train_cases=list_larn, 
           U=Lmat, s=lamda, VT=Rmat, 
           nparticle=len(list_larn), n_dim=30, iout=istep, 
           niters=nstep, ncheck_pnt=cpnt, dir_res=res_dir, 
           vald_txt=txt_vald)


if __name__ == '__main__':
    lg.basicConfig(filename=bf.makedirs('../res/{}/log'.format(batch)), 
                   level=lg.INFO)
    my_lg = lg.getLogger('my_logger')
    my_lg.setLevel(lg.DEBUG)
    my_lg.debug(time.strftime("%Y/%m/%d %H:%M:%S"))
    my_lg.debug('Looks like everything is off to a good start ;)')

    try:
        main(my_lg)
    except Exception:
        my_lg.exception("Fatal error in main loop")
