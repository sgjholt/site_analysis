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


#def vel_models(subset):
#
#    plt.step(subset.min(axis=0)[0:5], [x for x in range(1, 6)], '--k')
#    plt.step(subset.max(axis=0)[0:5], [x for x in range(1, 6)], '--k')
#    plt.step(subset.loc[0][0:5], [x for x in range(1, 6)], 'k', linewidth=2, label='Original')
#
#    plt.gca().invert_yaxis()
#    plt.xlabel('Vs [m/s]')
#    plt.ylabel('layer num')
#    plt.title('Model w/Misfit < Original ')
#    plt.title('Model w/Misfit < Original FKSH11')



