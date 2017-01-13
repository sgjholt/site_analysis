import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from site_class import Site


def rand_sites(sample_size):
    sites = sorted(pd.read_csv('/data/share/Japan/Kik_catalogue.csv', index_col=0).site.unique())
    return [sites[num] for num in random.sample(range(len(sites)-1), sample_size)]


def rand_qc():
    sample = [Site(s) for s in rand_sites(10)]
    for i, site in enumerate(sample):
        plt.subplot(2, 5, i+1)
        site.qc(plot_on=True)





