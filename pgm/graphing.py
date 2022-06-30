import matplotlib.pyplot as plt
plt.switch_backend("agg")
import matplotlib.colors as cm
import numpy as np
import seaborn as sns
import os

#function for calculating number of principal components to use:
def calc_num_components(cum_var, threshold):
    for i in range(cum_var.shape[0]):
        if cum_var[i] >= threshold:
            return i+1

def make_pf(step, elap_time, particles, weights, r_pos, xlim, ylim, save_dir):
    #sns.set()
    plt.figure(step, figsize=(3.5,3.5))
    plt.rcParams['font.size'] = 13
    ppx = particles[:,0]
    ppy = particles[:,1]
    ppz = weights
    idx = ppz.argsort()
    ppx, ppy, ppz = ppx[idx], ppy[idx], ppz[idx]
    sns.scatterplot(x=ppx, y=ppy,
        hue=ppz, hue_norm=(ppz.min(),ppz.max()),
        size=ppz, size_norm=(ppz.min(),ppz.max()),
        sizes=(35,200), alpha=0.8)
    plt.scatter(r_pos[0], r_pos[1], marker='o', facecolors='none', s=200, linewidths=1.5,
                color='red', alpha=.9, label='Test')

    plt.title('Time: {0:=6d} [s]'.format(elap_time))
    plt.xlabel(r'$\alpha_1$')
    plt.ylabel(r'$\alpha_2$')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.tight_layout()
    plt.legend(loc='best',fontsize=7)
    plt.savefig(save_dir, format='png',dpi=300)
    plt.clf()
    plt.close(step)


def alpha_comparison(i, case, mcase, most_case, 
                     tsta, tend, dsta, dend, 
                     A, A_dash, VT, n_dim, ncheck_pnt, 
                     dt, nrecord, elap_time, dir_res):

    wave_t = A[tsta:tend, :n_dim] @ VT[:n_dim, ncheck_pnt]
    wave_d = A_dash[dsta:dend, :n_dim] @ VT[:n_dim, ncheck_pnt]
    
    at = A[tsta:tend,:n_dim]
    ad = A_dash[dsta:dend,:n_dim]

    tt = np.arange(0, nrecord*dt, dt)

    #nalpha = [3, 4, 5, 6, 7, 8]
    nalpha = [0, 1, 2, 3, 4, 5]
    irow = len(nalpha) / 2 + 1
    icol = 2

    plt.figure(i, figsize=(6,8))
    plt.rcParams['font.size'] = 11
    plt.title('Results of update/estimation')


    for ii, ialpha in enumerate(nalpha):

      plt.subplot(irow, icol, ii+1)
      plt.plot(tt, ad[:,ialpha], lw=1.0, ls='solid', color='black')
      plt.plot(tt, at[:,ialpha], lw=0.75, ls='dashed', color='red')
      y_min, y_max = np.amin(ad[:,ialpha])*1.1, np.amax(ad[:,ialpha])*1.1
      plt.axvline(x=elap_time, ymin=y_min, ymax=y_max, c='blue', lw =0.5, ls='solid')
      plt.xlabel('Time [s]')
      plt.ylabel(r'$\alpha_{0:d}$'.format(ialpha+1))

    plt.subplot(irow, 1, irow)
    plt.plot(tt, wave_d[:,], lw=1.0, ls='solid', color='black', label='Test case({})'.format(case))
    plt.plot(tt, wave_t[:,], lw=0.75, ls='dashed', color='red', label='Most likely({})'.format(most_case))
    y_min, y_max = np.amin(wave_d[:,])*1.1, np.amax(wave_d[:,])*1.1
    plt.axvline(x=elap_time, ymin=y_min, ymax=y_max, c='blue', lw =0.5, ls='solid')
    plt.xlabel('Time [s]')
    plt.ylabel(r'wave height $\eta$ [m]')
    plt.legend(loc='best',fontsize=7)
    plt.title('Validation')

    plt.tight_layout()
    
    respath = os.path.join(dir_res, 'figures/res-pf{0:02d}-{1}.png'.format(i, case))
    plt.savefig(respath, format='png',dpi=300)
    plt.clf()
    plt.close(i)

def make_scree(s, threshold, save_dir):
    s = s*(1/np.sum(s))
    cumulative_var = np.cumsum(np.round(s, decimals=3)*100)
    n_components = calc_num_components(cumulative_var, threshold)
    cumulative_var = cumulative_var[:int(n_components*5/4)]

    n = len(cumulative_var)
    y_vals = [num for num in cumulative_var]
    x_vals = [num for num in range(1,n+1)]
    
    colors = ['lightcoral' if x == n_components else 'gainsboro' for x in x_vals]
    fig, ax = plt.subplots(figsize=(7.0,2.0))
    ax.grid(False)
    #ax.set_title('Principal Components Cumulative Variance')
    ax.set_ylabel('Cumulative Variance [%]', fontsize=8)
    ax.set_xlabel('Number of modes $r$', fontsize=8)

    # threshold var
    ax.axhline(threshold, color='black', linewidth=0.5, ls='dashed');
    sns.barplot(x=x_vals, y=y_vals, ax=ax, palette=colors, alpha=0.3)    

    ax.set_xticklabels(np.arange(5, 40, 5))
    ax.set_xticks(np.arange(4, 39, 5))
    ax.tick_params(axis='both', which='major', labelsize=8)
    fig.tight_layout()
    fig.savefig(save_dir)
    plt.clf()
    plt.close('all')

def make_wave(ndata, vec_solid, vec_dot, n_cases, n_modes, title, save_dir):

    x = np.linspace(0, ndata*5, ndata)

    plt.figure(figsize=(3.8,2.0))

    rmse = (vec_solid - vec_dot) * (vec_solid - vec_dot)
    rmse = np.mean(rmse)

    plt.plot(x, vec_solid, label='original', color='black', ls='solid', lw=1.0)
    plt.plot(x, vec_dot, label='reconstructed', color='orangered', ls='dashed', lw=0.3)
    #plt.title(title,fontsize=6)
    plt.title('$r$ = {}'.format(n_modes),fontsize=7)
    plt.legend(loc='best', fontsize=5)
    plt.ylim([-8.5, 8.5])
    plt.ylabel('$\eta$ [m]', fontsize=6)
    plt.xlabel('Time [s]', fontsize=6)
    plt.tight_layout()

    plt.savefig(save_dir)
    plt.clf()
    plt.close('all')
    return rmse

