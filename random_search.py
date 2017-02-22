__author__ = 'James Holt'

# random_search - script to run random velocity profile searches
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
    iterations = 1000
    name = 'random_uniform_'+site+'_'+str(iterations)
    wd = '/data/share/Japan/SiteInfo/S_B/Repi_lessthan_300_depth_lessthan_25/'
    rd = '/data/share/Japan/SiteInfo/S_B/{0}_Vs_MC'.format(site)
    if not os.path.isdir(rd):
        os.mkdir(rd)
    uniform_search(site, wd, rd, 25, 100, iterations, name, (0.4, 0.6), 1, 0, (0, 25), False, False)


def uniform_search(*args):
    site, wd, rd, pct, steps, iters, name, weights, lam, i_ang, x_range, elastic, cadet_correct = args
    names = ['site', 'SB_dir', 'run_dir', 'pct_variation', 'model_steps', 'iterations', 'misfit_weights', 'lambda',
             'i_ang', 'x_range_for_xcor', 'elastic', 'cadet_correct']
    vrs = [site, wd, rd, pct, steps, iters, weights, lam, i_ang, x_range, elastic, cadet_correct]

    with open(rd+'/'+name+'.cfg', 'wt') as f:
        f.write('config file for {0}'.format(name).capitalize())
        f.write('\n')
        f.write('\n')
        for i, name in enumerate(names):
            f.write("{0} = {1}".format(name, vrs[i]))
            f.write('\n')

    site = sd.Sim1D(site, working_directory=wd, run_dir=rd)

    site.uniform_random_search(pct_variation=pct, steps=steps, iterations=iters, name=name, weights=weights, lam=lam,
                               i_ang=i_ang, x_cor_range=x_range, elastic=elastic, cadet_correct=cadet_correct,
                               save=True)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- {0} seconds elapsed ---".format(time.time() - start_time))
