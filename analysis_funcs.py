from site_class import Site
from utils import rand_sites
import matplotlib.pyplot as plt


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


def vel_model_range(orig, subset, thresh, site, pct_v):
    """

    :param orig:
    :param subset:
    :param thresh:
    :param site:
    :param pct_v:
    :return:
    """
    layers = [x for x in range(1, len(orig.loc[0][0:-4])+1)]

    plt.figure()

    plt.step(orig.loc[0][0:-4], layers, 'k', linewidth=2, label='orig')
    plt.step(orig.loc[0][0:-4] * (1-pct_v/100), layers, 'r', linestyle='dashed', label='model range')
    plt.step(orig.loc[0][0:-4] * (1+pct_v/100), layers, 'r', linestyle='dashed')
    plt.step(subset.min(axis=0)[0:-4], layers, 'k', label='range < tot_mis={}'.format(thresh))
    plt.step(subset.max(axis=0)[0:-4], layers, 'k')

    plt.gca().invert_yaxis()
    plt.xlabel('Vs [m/s]')
    plt.ylabel('layer num')
    plt.title('Model w/Misfit < Original ')
    plt.title('Model w/Misfit < Original {}'.format(site))
    plt.legend()
    plt.show()


def best_fitting_model(site_obj, orig, minimum=None, thrsh=None, elastic=False, cadet_correct=False):
    _freqs = site_obj.sb_ratio().columns.values.astype(float)  # str by default
    freqs = (round(float(_freqs[0]), 2), round(float(_freqs[-1]), 2), len(_freqs))
    if minimum is not None:
        subset = orig.query('total_mis == {0}'.format(orig.total_mis.min()))
        model = subset.loc[subset.index[0]][0:-3]
        site_obj.modify_site_model(model)
        site_obj.elastic_forward_model(elastic=elastic, plot_on=True, freqs=freqs)
        site_obj.plot_sb(stdv=(1,), cadet_correct=cadet_correct, show=True)

    if thrsh is not None:
        subset = orig.query('total_mis <= {0}'.format(thrsh))
        for row in subset.iterrows():
            model = [num[1] for num in row[1].iteritems()]
            site_obj.modify_site_model(model[0:-3])
            fwd = site_obj.elastic_forward_model(elastic=elastic, freqs=freqs)
            plt.loglog(site_obj.Amp['Freq'], fwd, label='mis={}'.format(round(model[-1], 3)))
        site_obj.plot_sb(stdv=(1,), cadet_correct=cadet_correct, show=True)
