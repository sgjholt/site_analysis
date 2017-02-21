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

