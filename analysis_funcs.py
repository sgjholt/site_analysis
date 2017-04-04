from site_class import Site
# from find_peaks import detect_peaks
from utils import rand_sites, fill_troughs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob


def rand_qc(working_dir):
    """
    Random sample of my site specific S/B ratios plotted against Ben's for quality control.

    :param working_dir: string object containing full path to S/B ratio files
    :return: 10 sites selected at random as a plot (if QC is available for those sites)
    """
    sample = [Site(s, working_dir) for s in rand_sites(10)]
    for i, site in enumerate(sample):
        plt.subplot(2, 5, i + 1)
        site.qc(plot_on=True)
    plt.show()


def vel_model_range(orig, subset, thresh, site_obj, pct_v, save=False, user='sgjholt', dpi=None):
    """

    :param orig:
    :param subset:
    :param thresh:
    :param site:
    :param pct_v:
    :return:
    """
    layers = [x for x in range(1, len(orig.loc[0][0:-5]) + 1)]

    plt.figure(figsize=(8, 13))

    plt.step(orig.loc[0][0:-5], layers, 'k', linewidth=2, label='orig')
    plt.step(orig.loc[0][0:-5] * (1 - pct_v / 100), layers, 'r', linestyle='dashed', label='model range')
    plt.step(orig.loc[0][0:-5] * (1 + pct_v / 100), layers, 'r', linestyle='dashed')
    plt.step(subset.min(axis=0)[0:-5], layers, 'b', label='range < tot_mis={}'.format(thresh))
    plt.step(subset.max(axis=0)[0:-5], layers, 'b')

    plt.gca().invert_yaxis()
    plt.xlabel('Vs [m/s]')
    plt.ylabel('layer num')
    plt.title('Model w/Misfit < Original {0}\n Sub-Layers: {1}'.format(site_obj.site,
                                                                       orig.n_sub_layers.values[0].astype(int)))
    plt.legend()

    if save:
        plt.savefig(
            '/home/{3}/plots/synthetic_tests/{0}-MC-Vel-Model-{1}-iterations-Misfit-loe{2}-Sub-layers-{4}.pdf'.format(
                site_obj.site, len(orig) - 1, thresh, user, orig.n_sub_layers[0].astype(int)), dpi=dpi, facecolor='w',
            edgecolor='w', orientation='portrait', format='pdf', transparent=False, bbox_inches=None,
            pad_inches=0.1, frameon=None)
    else:
        plt.show()


def best_fitting_model(site_obj, orig, mis='total_mis',minimum=None, thrsh=None, elastic=False, cadet_correct=False,
                       fill_troughs_pct=None, sub_layers=True, save=False, dpi=None, user='sgjholt', subplots=False):

    _freqs = site_obj.sb_ratio().columns.values.astype(float)  # str by default
    freqs = (round(float(_freqs[0]), 2), round(float(_freqs[-1]), 2), len(_freqs))

    if not subplots:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(1, 1, 1)



    if minimum is not None:
        subset = orig.query('{0} == {1}'.format(mis, orig.total_mis.min()))
        model = subset.loc[subset.index[0]][0:-4].values
        site_obj.modify_site_model(model, sub_layers=sub_layers)
        if fill_troughs_pct is not None:
            site_obj.GenFreqAx(freqs[0], freqs[1], freqs[2])
            plt.plot(freqs, fill_troughs(site_obj.elastic_forward_model(elastic=elastic)[::, 0],
                                         pct=fill_troughs_pct))
        else:
            site_obj.elastic_forward_model(elastic=elastic, plot_on=True)

        site_obj.plot_sb(stdv=(1,), cadet_correct=cadet_correct)

    if thrsh is not None:
        subset = orig.query('{0} <= {1}'.format(mis, thrsh))
        for row in subset.iterrows():
            model = np.array([num[1] for num in row[1].iteritems()])
            site_obj.modify_site_model(model[0:-4], sub_layers=sub_layers)
            if fill_troughs_pct is not None:
                fwd = fill_troughs(site_obj.elastic_forward_model(elastic=elastic, freqs=freqs)[::, 0],
                                   pct=fill_troughs_pct)
            else:
                fwd = site_obj.elastic_forward_model(elastic=elastic)
            plt.loglog(site_obj.Amp['Freq'], fwd, label='mis={}'.format(round(model[-2], 5)))
        site_obj.plot_sb(stdv=(1,), cadet_correct=cadet_correct)
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=0.7)
    plt.title('{0}: MC - {1} - iterations - Misfit <= {2} - Sub-layers: {3}'.format(site_obj.site, len(orig) - 1, thrsh,
                                                                                    orig.n_sub_layers[0].astype(int)))
    plt.axis('tight')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', lw=2)
    ax.set_xticks([0.1, 0.5, 1, 5, 10, 15, 20, 25])

    if save:
        plt.savefig('/home/{3}/plots/synthetic_tests/{0}-MC-{1}-iterations-Misfit-loe{2}-Sub-layers-{4}.pdf'.format(
            site_obj.site, len(orig) - 1, thrsh, user, orig.n_sub_layers[0].astype(int)), dpi=dpi, facecolor='w',
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
    site_obj.elastic_forward_model(elastic=False, plot_on=True, show=False)
    mods = [site_obj.Mod['Qs']]
    for i in range(its+1):
        randomise_q(site_obj)
        site_obj.elastic_forward_model(elastic=False, plot_on=True, show=False)
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
    site_obj.elastic_forward_model(elastic=False, plot_on=True)
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
    dat = site_obj.elastic_forward_model(elastic=False)[::, 0]
    plt.figure(1)
    for pct in fill_pct:
        plt.plot(site_obj.Amp['Freq'], fill_troughs(dat, pct), label='{0} pct fill'.format(pct))
    plt.grid(which='both')
    plt.legend()
    site_obj.plot_sb(stdv=(1,))
    site_obj.elastic_forward_model(elastic=False, plot_on=True)


