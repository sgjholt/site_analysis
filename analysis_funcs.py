import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from site_class import Site
from utils import rand_sites, fill_troughs, exp_cdf


def qc_sb(working_dir, sites=None, random=False):
    """
    Random sample of my site specific S/B ratios plotted against Ben's for quality control.

    :param working_dir: string object containing full path to S/B ratio files
    :param sites
    :param random
    :return: 10 sites selected at random as a plot (if QC is available for those sites)
    """
    if random:
        sample = [Site(s, working_dir) for s in rand_sites(6)]
    else:
        if sites is not None:
            sample = [Site(s, working_dir) for s in sites]
        else:
            print('Please give list of sites')
            return

    fig = plt.figure(figsize=(10, 10))
    font = {'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', lw=2)
    subplots = []
    for i, site in enumerate(sample):
        if i == 0:
            ax1 = fig.add_subplot(2, 3, 1)
            site.qc(plot_on=True)
            ax1.set_xticks([0.1, 1, 5, 10, 25])
            ax1.grid(which='minor', alpha=0.5)
            ax1.grid(which='major', alpha=0.7)
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
            ax1.xaxis.label.set_visible(False)
            plt.ylim([0.1, 100])
        else:
            subplots.append(fig.add_subplot(2, 3, i + 1, sharex=ax1, sharey=ax1))
            site.qc(plot_on=True)
            plt.ylim([0.1, 100])

    for i, subplot in enumerate(subplots):
        subplot.set_xticks([0.1, 1, 5, 10, 25])
        subplot.grid(which='minor', alpha=0.5)
        subplot.grid(which='major', alpha=0.7)
        subplot.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
        subplot.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
        if i in [0, 1]:
            subplot.xaxis.label.set_visible(False)
            subplot.yaxis.label.set_visible(False)
        if i in [3, 4]:
            subplot.yaxis.label.set_visible(False)
    # plt.tight_layout()
    plt.show()


def vel_model_range(site_obj, orig, thrsh=(), save=False, user='sgjholt', dpi=None):
    """
    :param site_obj
    :param orig:
    :param thrsh:
    :param save:
    :param user:
    :param dpi:
    :return:
    """

    orig_c = orig.copy(deep=True)

    orig_c.freq_mis = exp_cdf(orig_c.freq_mis.apply(np.abs), 1)  # apply normalisation (f-lag)
    # apply normalisation (rms)
    orig_c.amp_mis = (orig_c.amp_mis - orig_c.amp_mis.min()) / (orig_c.amp_mis.max() - orig_c.amp_mis.min())

    pct_v = int(site_obj.sim_pars['pct_variation'])

    # subset = orig.apply(np.abs)
    subset = orig_c[orig_c.freq_mis <= orig_c.freq_mis[0] * thrsh[1]]
    subset = subset[subset.amp_mis <= orig_c.amp_mis[0] * thrsh[0]]
    print('{0} models found'.format(len(subset)))

    layers = [x - 0.5 for x in range(1, len(orig_c.loc[0][0:-4]) + 1)]
    layers.append(layers[-1] + 1)

    fig, ax = plt.subplots(figsize=(8, 13))
    # repeat half-space velocity so the step plot can show variation in half-space layer clearly
    vels = np.array(orig_c.loc[0][0:-4].values.tolist() + [orig_c.loc[0][0:-4].values[-1]])

    ax.step(vels, layers, 'k', linewidth=2, label='orig')
    ax.step(vels * (1 - pct_v / 100), layers, 'r', linestyle='dashed', label='model range')
    ax.step(vels * (1 + pct_v / 100), layers, 'r', linestyle='dashed')
    ax.step(subset.min(axis=0)[0:-4].values.tolist() + [subset.min(axis=0)[0:-4].values[-1]], layers, 'b',
            label='range < tot_mis={}'.format((round(
                orig_c.amp_mis[0] * thrsh[0], 3), round(orig_c.freq_mis[0] * thrsh[1], 3))))
    ax.step(subset.max(axis=0)[0:-4].values.tolist() + [subset.max(axis=0)[0:-4].values[-1]], layers, 'b')
    plt.ylim([0.5, layers[-2] + 1])
    plt.gca().invert_yaxis()
    plt.xlabel('Vs [m/s]')
    plt.ylabel('layer num')
    plt.title('Model w/Misfit < Original {0}\n Sub-Layers: {1}'.format(site_obj.site,
                                                                       orig_c.n_sub_layers.values[0].astype(int)))
    plt.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if save:
        plt.savefig(
            '/home/{3}/plots/synthetic_tests/{0}-MC-Vel-Model-{1}-iterations-Misfit-loe{2}-Sub-layers-{4}.pdf'.format(
                site_obj.site, len(orig) - 1, thrsh, user, orig_c.n_sub_layers[0].astype(int)), dpi=dpi, facecolor='w',
            edgecolor='w', orientation='portrait', format='pdf', transparent=False, bbox_inches=None,
            pad_inches=0.1, frameon=None)
    else:
        plt.show()


def best_fitting_model(site_obj, orig, minimum=None, thrsh=None, elastic=False, cadet_correct=False,
                       fill_troughs_pct=None, sub_layers=True, save=False, dpi=None, user='sgjholt', subplots=False,
                       motion='outcrop', konno_ohmachi=None):
    orig_c = orig.copy(deep=True)

    orig_c.freq_mis = exp_cdf(orig_c.freq_mis.apply(np.abs), 1)  # apply normalisation (f-lag)
    # apply normalisation (rms)
    orig_c.amp_mis = (orig_c.amp_mis - orig_c.amp_mis.min()) / (orig_c.amp_mis.max() - orig_c.amp_mis.min())

    site_obj.reset_site_model()

    _freqs = site_obj.sb_ratio().columns.values.astype(float)  # str by default

    if not subplots:
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(1, 1, 1)
        site_obj.linear_forward_model_1d(elastic=False, plot_on=True, motion=motion, konno_ohmachi=konno_ohmachi)
        font = {'weight': 'bold',
                'size': 18}

        matplotlib.rc('font', **font)
        matplotlib.rc('lines', lw=2)

    if minimum is not None:
        # find all models better(or equal to) original model - make sure absolute value of f_lag considered
        # subset = orig.apply(np.abs)
        subset = orig_c[orig_c.freq_mis <= orig_c.freq_mis.min()]
        subset = subset[subset.amp_mis <= subset.amp_mis.min()]

        model = subset.loc[subset.index[0]][0:-3].values
        site_obj.modify_site_model(model, sub_layers=sub_layers)
        if fill_troughs_pct is not None:
            plt.plot(_freqs, fill_troughs(site_obj.linear_forward_model_1d(elastic=elastic, motion=motion,
                                                                           konno_ohmachi=konno_ohmachi),
                                          pct=fill_troughs_pct))
        else:
            plt.plot(site_obj.Amp['Freq'], site_obj.linear_forward_model_1d(elastic=elastic, motion=motion,
                                                                            konno_ohmachi=konno_ohmachi),
                     label='SHTF - Optimised')

        site_obj.plot_sb(stdv=(1,), cadet_correct=cadet_correct)

    if thrsh is not None:

        #  subset = orig.apply(np.abs)
        subset = orig_c[orig_c.freq_mis <= orig_c.freq_mis[0] * thrsh[1]]
        subset = subset[subset.amp_mis <= orig_c.amp_mis[0] * thrsh[0]]
        print('{0} models found'.format(len(subset)))
        for row in subset.iterrows():
            model = np.array([num[1] for num in row[1].iteritems()])
            site_obj.modify_site_model(model[0:-3], sub_layers=sub_layers)
            if fill_troughs_pct is not None:
                fwd = fill_troughs(site_obj.linear_forward_model_1d(elastic=elastic, motion=motion,
                                                                    konno_ohmachi=konno_ohmachi),
                                   pct=fill_troughs_pct)
            else:
                fwd = site_obj.linear_forward_model_1d(elastic=elastic, motion=motion, konno_ohmachi=konno_ohmachi)
            plt.plot(site_obj.Amp['Freq'], fwd, label='amp_mis={0}, freq_mis={1}'.format(np.round(model[-3], 5),
                                                                                         np.round(model[-2], 5)))
        site_obj.plot_sb(stdv=(1,), cadet_correct=cadet_correct)

    if minimum is not None:
        mfit = np.round(orig_c.amp_mis.min(), 4), np.round(orig_c.freq_mis.min(), 4)
    if thrsh is not None:
        mfit = np.round(orig_c.amp_mis[0] * thrsh[0], 2), np.round(orig_c.freq_mis[0] * thrsh[1], 4)

    ax.set_xticks([0.1, 1, 5, 10, 15, 20, 25])
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=0.7)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))

    plt.title(
        '{0}: MC - {1} - iterations - Misfit <= {2} - Sub-layers: {3}'.format(site_obj.site, len(orig_c) - 1, mfit,
                                                                              orig_c.n_sub_layers[0].astype(int)))
    plt.axis('tight')

    plt.legend(loc=2)

    if save:
        plt.savefig('/home/{4}/plots/synthetic_tests/{0}-MC-{1}-iterations-Misfit-loe{2}-{3}-Sub-layers-{5}.pdf'.format(
            site_obj.site, len(orig_c) - 1, thrsh[0], thrsh[1], user, orig.n_sub_layers[0].astype(int)), dpi=dpi,
            facecolor='w',
            edgecolor='w', orientation='portrait', format='pdf', transparent=False, bbox_inches=None,
            pad_inches=0.1, frameon=None)
    else:
        plt.show()


# def plot

def randomise_q(site_obj):
    """


    :param site_obj: site object
    :return: None - inplace function
    """
    site_obj.Mod['Qs'] = [0 for _ in site_obj.Mod['Qs']]
    site_obj.Mod['Qs'] = (np.array(site_obj.Mod['Qs'])+np.random.randint(2, 50, len(site_obj.Mod['Qs']))).tolist()


def search_qs(site_obj, its):
    site_obj.Mod['Qs'] = [10 for _ in site_obj.get_velocity_profile()['vs']]
    site_obj.linear_forward_model_1d(elastic=False, plot_on=True, show=False)
    mods = [site_obj.Mod['Qs']]
    for i in range(its+1):
        randomise_q(site_obj)
        site_obj.linear_forward_model_1d(elastic=False, plot_on=True, show=False)
        print('{0}'.format(site_obj.Mod['Qs']))
        mods.append([str(num) for num in site_obj.Mod['Qs']])
    plt.legend([mod for mod in mods])


def compare_strata(site_obj):
    files = glob.glob('/home/sgjholt/Compare_strata/*.csv')
    all_data = [np.loadtxt(fname, skiprows=3, delimiter=',') for fname in files]
    #return all_data
    plt.figure()
    ax1 = plt.subplot(121)
    plt.xlabel('Freq [Hz]')
    plt.ylabel('SB Ratio')
    plt.grid(which='both')
    for i, data in enumerate(all_data):
        if i > 4:
            plt.legend()
            plt.subplot(122)
        plt.loglog(data[:, 0], data[:, 1], label='{0}'.format(files[i].split(' m ')[-1].split('-')[0]))
    site_obj.linear_forward_model_1d(elastic=False, plot_on=True)
    site_obj.plot_sb(stdv=(1,))
    plt.xlabel('Freq [Hz]')
    plt.ylabel('SB Ratio')
    plt.legend()
    plt.grid(which='both')

    #for i in range(1, len(all_data)):
    #    all_data[0] = np.hstack((all_data[0], all_data[i][:, 1:]))
    #ax2 = plt.subplot(122, sharey=ax1)
    #plt.loglog(all_data[0][:, 0], np.exp(np.mean(np.log(all_data[0][:, 1:]), axis=1)), label='mean strata')
    #plt.plot()
    #site_obj.plot_sb(stdv=(1,))
    #plt.xlabel('Freq [Hz]


def ramp_func(x, a=0, b=1, c=0.5, r=15):
    """

    :param x:
    :param a:
    :param b:
    :param c:
    :param r:
    :return:
    """
    return (a*np.exp(c*r) + b * np.exp(r*x)) / (np.exp(c*r) + np.exp(r*x))


def plotr():
    x = np.arange(0, 3+0.01, 0.01)
    r=0
    for i in range(1, 6):
        r += i*2
        plt.plot(x, ramp_func(x, r=r), label='r={0}'.format(int(r)))
    plt.xlabel('$f$')
    plt.ylabel('$A$')
    plt.legend()
    plt.title('Ramp Func')
    plt.xlim([-0.1, 3])
    plt.ylim([0, 1.1])
    plt.grid(which='both')


def compare_fill(site_obj, fill_pct=()):
    dat = site_obj.linear_forward_model_1d(elastic=False)
    plt.figure(1)
    for pct in fill_pct:
        plt.plot(site_obj.Amp['Freq'], fill_troughs(dat, pct), label='{0} pct fill'.format(pct))
    plt.grid(which='both')
    plt.legend()
    site_obj.plot_sb(stdv=(1,))
    site_obj.linear_forward_model_1d(elastic=False, plot_on=True)


# Only works to plot strata data files - oc2within v oc2oc

def plot_comp_strata(site_obj, path):
    font = {'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', lw=2)
    data = np.loadtxt(path, delimiter=',')
    df = site_obj.sb_ratio()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.loglog(df.columns.values.astype(float), np.exp(df.loc['mean']), 'k', label='S/B Ratio')
    plt.loglog(data[:, 0], data[:, 1], 'b', label='OC-WN')
    plt.loglog(data[:, 0], data[:, 2], 'r', label='OC-OC')
    plt.hlines(1, data[:, 0][0], data[:, 0][-1], colors='k', linestyles='dashed')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('SHTF')
    plt.title(site_obj.site+': 1D-SHTF')
    ax.set_xticks([0.1, 1, 10, 25])
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=0.7)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    plt.ylim([0.5, 15])
    plt.xlim([0.1, 25])
    plt.legend(loc=2)


def plot_misfit_space(table):
    # amp_normed = table.amp_mis.values/np.sqrt(np.trapz(table.amp_mis.values**2))
    amp_normed = (table.amp_mis - table.amp_mis.min()) / (table.amp_mis.max() - table.amp_mis.min())
    xcor_normed = exp_cdf(np.abs(table.freq_mis.values), lam=1)

    def onpick(event):
        models = []
        index = event.ind
        for ind in index:
            print('Trial: {0}-RMS:{1}-F_Lag:{2}'.format(ind, np.take(amp_normed, ind), np.take(xcor_normed, ind)))
            print('Model: {0}'.format(table.loc[int(ind)].values[:-3]))
            print('\n')

    fig, ax = plt.subplots()
    font = {'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', lw=2)
    ax.scatter(amp_normed, xcor_normed, picker=True, label='Random Trials')
    ax.hlines(1, amp_normed.min(), amp_normed.max(), linestyles='dashed', colors='red', label='Auto-Rejected Models')
    ax.hlines(xcor_normed[0], amp_normed.min(), amp_normed.max(), linestyles='dashed', colors='blue')
    ax.vlines(amp_normed[0], xcor_normed.min(), xcor_normed.max(), linestyles='dashed', colors='blue')
    ax.scatter(amp_normed[0], xcor_normed[0], s=40, c='red', label='Initial Model')
    plt.xlabel('$RMS$ $Normalised$')
    plt.ylabel('$Frequency$ $Lag$ $Normalised$')
    plt.title('Misfit Space:')
    plt.ylim([-0.1, 1.05])
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.legend(loc=1)
    # plt.xlim([0, amp_normed.max()+0.1])
    plt.show()
