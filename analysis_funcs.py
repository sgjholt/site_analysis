from site_class import Site
from utils import rand_sites
import matplotlib.pyplot as plt


def rand_qc(working_dir):
    sample = [Site(s, working_dir) for s in rand_sites(10)]
    for i, site in enumerate(sample):
        plt.subplot(2, 5, i + 1)
        site.qc(plot_on=True)
