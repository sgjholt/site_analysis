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
import sys

__author__ = 'James Holt'
# ---------------------------------modules--------------------------------------#


def main():
    site = sys.argv[1]
    iterations = int(sys.argv[2])
    name = site+'_rnd_ufm_'+'_'+str(iterations)+'_run_0'
    wd = '/data/share/Japan/SiteInfo/S_B/Repi_lessthan_300_depth_lessthan_25/'
    rd = '/data/share/Japan/SiteInfo/S_B/{0}_Vs_MC_subl/'.format(site)

    if not os.path.isdir(rd):  # if not a directory, make it
        print('creating dir: {0}'.format(rd))
        os.mkdir(rd)

    if os.path.exists(rd+name+'.cfg'):
        j = 0
        while os.path.exists(rd+name+'.cfg'):
            j += 1  # if the file already exists, give it another name
            name = name.replace('run_0', 'run_{0}'.format(j))
        print('name changed to: {0}'.format(name))

    sub_uniform_search(site, wd, rd, 250.0, 10, iterations, name, 0, (0.5, 25), 100, False, False, None,
                       (1, 2, 3), 'outcrop')


def sub_uniform_search(*args):
    site, wd, rd, pct, steps, iters, name, i_ang, x_range, const_q, elastic, cadet_correct, \
    fill_troughs, n_sub_layers, motion = args
    titles = ['site', 'SB_dir', 'run_dir', 'pct_variation', 'model_steps', 'iterations',
              'i_ang', 'x_range_for_xcor', 'const_q', 'elastic', 'cadet_correct', 'fill_troughs', 'n_sub_layers',
              'motion']
    vrs = [site, wd, rd, pct, steps, iters, i_ang, x_range, const_q, elastic, cadet_correct, fill_troughs,
           n_sub_layers, motion]

    with open(rd+name+'.cfg', 'wt') as f:
        f.write('config file for {0}'.format(name).upper())
        f.write('\n')
        f.write('\n')
        for i, title in enumerate(titles):
            f.write("{0} = {1}".format(title, vrs[i]))
            f.write('\n')

    site = sd.Sim1D(site, working_directory=wd, run_dir=rd)

    site.uniform_sub_random_search(pct_variation=pct, steps=steps, iterations=iters, name=name,
                                   i_ang=i_ang, x_cor_range=x_range, const_q=const_q, elastic=elastic,
                                   cadet_correct=cadet_correct, fill_troughs_pct=fill_troughs, save=True,
                                   n_sub_layers=n_sub_layers, motion=motion)


#def uniform_search(*args):
#    """
#    UNFINISHED
#    :param args:
#    :return:
#    """
#    site, wd, rd, pct, steps, iters, name, weights, lam, i_ang, x_range, const_q, elastic, cadet_correct, \
#    fill_troughs, n_sub_layers = args
#    titles = ['site', 'SB_dir', 'run_dir', 'pct_variation', 'model_steps', 'iterations', 'misfit_weights', 'lambda',
#              'i_ang', 'x_range_for_xcor', 'const_q', 'elastic', 'cadet_correct', 'fill_troughs', 'n_sub_layers']
#    vrs = [site, wd, rd, pct, steps, iters, weights, lam, i_ang, x_range, const_q, elastic, cadet_correct, fill_troughs,
#           n_sub_layers]

#    with open(rd + name + '.cfg', 'wt') as f:
#        f.write('config file for {0}'.format(name).upper())
#        f.write('\n')
#        f.write('\n')
#        for i, title in enumerate(titles):
#            f.write("{0} = {1}".format(title, vrs[i]))
#            f.write('\n')

#    site = sd.Sim1D(site, working_directory=wd, run_dir=rd)

#    site.uniform_sub_random_search(pct_variation=pct, steps=steps, iterations=iters, name=name, weights=weights,
#                                   lam=lam,
#                                   i_ang=i_ang, x_cor_range=x_range, const_q=const_q, elastic=elastic,
#                                   cadet_correct=cadet_correct, fill_troughs_pct=fill_troughs, save=True,
#                                   n_sub_layers=n_sub_layers)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- {0} seconds elapsed ---".format(time.time() - start_time))


