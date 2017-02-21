__author__ = 'James Holt'

# site_class - Base Site class object - handles site metadata - produces plots and other useful helper functions.
# VERSION: 0.9
# AUTHOR(S): JAMES HOLT - UNIVERSITY OF LIVERPOOL
#
# EMAIL: j.holt@liverpool.ac.uk
#
#
# ---------------------------------modules--------------------------------------#
import sim1D as sd
import time
import os
# ---------------------------------modules--------------------------------------#


def main():
    site = 'FKSH16'
    iterations=1000
    name = 'random_uniform_'+site+str(iterations)
    wd = '/data/share/Japan/SiteInfo/S_B/Repi_lessthan_300_depth_lessthan_25/'
    rd = '/data/share/Japan/SiteInfo/S_B/{0}_Vs_MC'.format(site)
    if not os.path.isdir(rd):
        os.mkdir(rd)
    if os.path.isfile(rd+'/'+name):
        name.join('1')
    uniform_search(site, 25, 100, iterations, name, (0.4, 0.6), 1, 0, (0, 25), True, False)


def uniform_search(*args):
    site, wd, rd, pct, steps, iters, name, weights, lam, i_ang, x_range, elastic, cadet_correct  = args

    site = sd.Sim1D(site, working_directory=wd, run_dir=rd)

    site.uniform_random_search(pct_variation=pct, steps=steps, iterations=iters, name=name, weights=weights, lam=lam,
                               i_ang=i_ang, x_cor_range=x_range, elastic=elastic, cadet_correct=cadet_correct, save=True)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- {0} seconds elapsed ---".format(time.time() - start_time))
