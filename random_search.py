# random_search - script to run random velocity profile searches
# VERSION: 0.9
# AUTHOR(S): JAMES HOLT - UNIVERSITY OF LIVERPOOL
#
# EMAIL: j.holt@liverpool.ac.uk
#
#
# ---------------------------------modules--------------------------------------#
import time
import datetime
import os
import sys
import numpy as np
import sim1D as sd

__author__ = 'James Holt'
# ---------------------------------modules--------------------------------------#


def main():
    """
    
    :return: 
    """
    try:
        site = sys.argv[1]
        iterations = int(sys.argv[2])
    except IndexError:
        print('Minimum input arguments not given: need site name and # of iters.')
        print('Exiting: no tasks performed.')
        sys.exit()

    try:
        motion = str(sys.argv[3])
        konno_ohmachi = int(sys.argv[4])
        rect_uni = str(sys.argv[5])
    except IndexError:
        motion = 'outcrop'
        konno_ohmachi = None
        rect_uni = 'uniform'

    name = site + '_' + rect_uni + '_' + str(iterations) + '_run_0_'
    wd = '/data/share/Japan/SiteInfo/S_B/VLER10_blind/'
    # rd = '/data/share/Japan/SiteInfo/S_B/{0}_Vs_MC_subl_{1}_smooth-{2}/'.format(site, motion, konno_ohmachi)
    rd = '/data/share/Japan/SiteInfo/S_B/VLER10_blind/'
    # TODO: Revert changes after Valerio blind test_cor_vs_space.py

    if not os.path.isdir(rd):  # if not a directory, make it
        print('creating dir: {0}'.format(rd))
        os.mkdir(rd)

    if os.path.exists(rd+name+'.cfg'):
        j = 0
        while os.path.exists(rd+name+'.cfg'):
            j += 1  # if the file already exists, give it another name
            name = name.replace('run_0', 'run_{0}'.format(j))
        print('name changed to: {0}'.format(name))
    if rect_uni == 'uniform':
        sub_uniform_search(site, wd, rd, 50, 10, iterations, name, 0, (0, 25), None, False, False, None,
                           (1, 5, 9), motion, konno_ohmachi)
    else:
        rect_space_search(site, wd, rd, 70, 3500, 20, iterations, name, 0, 2, True, (0, 25), None, False, False, None,
                          True, False, motion, None, (250, np.exp(1)))


def sub_uniform_search(*args):
    site, wd, rd, pct, steps, iters, name, i_ang, x_range, const_q, elastic, cadet_correct, \
    fill_troughs, n_sub_layers, motion, konno_ohmachi = args
    titles = ['site', 'SB_dir', 'run_dir', 'pct_variation', 'model_steps', 'iterations',
              'i_ang', 'x_range_for_xcor', 'const_q', 'elastic', 'cadet_correct', 'fill_troughs', 'n_sub_layers',
              'motion', 'konno_ohmachi_smoothing(b)']

    if const_q is None:
        const_q = 'Valerio-ETH-model'

    vrs = [site, wd, rd, pct, steps, iters, i_ang, x_range, const_q, elastic, cadet_correct, fill_troughs,
           n_sub_layers, motion, konno_ohmachi]

    with open(rd+name+'.cfg', 'wt') as f:
        f.write('config file for {0}'.format(name).upper())
        f.write('\n')
        f.write('\n')
        for i, title in enumerate(titles):
            f.write("{0} = {1}".format(title, vrs[i]))
            f.write('\n')

    if type(const_q) is str:
        const_q = None

    site = sd.Sim1D(site, working_directory=wd, run_dir=rd)

    site.uniform_sub_random_search(pct_variation=pct, steps=steps, iterations=iters, name=name,
                                   i_ang=i_ang, x_cor_range=x_range, const_q=const_q, elastic=elastic,
                                   cadet_correct=cadet_correct, fill_troughs_pct=fill_troughs, save=True,
                                   n_sub_layers=n_sub_layers, motion=motion, konno_ohmachi=konno_ohmachi)


def rect_space_search(*args):
    site, wd, rd, = args[0:3]
    name = args[7]

    titles = ['site', 'data_dir', 'write_dir', 'low', 'high', 'model_steps', 'iterations', 'name',
              'i_ang', 'spacing[m]', 'force_min_spacing', 'x_range_for_xcor', 'const_q', 'elastic', 'cadet_correct',
              'fill_troughs', 'save',
              'debug', 'motion', 'konno_ohmachi_smoothing(b)', 'log_spacing-base']

    const_q = args[9]
    if const_q is None:
        const_q = 'Valerio-ETH-model'

    with open(rd + name + '.cfg', 'wt') as f:
        f.write('config file for {0}'.format(name).upper())
        f.write('\n')
        f.write('\n')
        for i, var in enumerate(args):
            f.write("{0} = {1}".format(titles[i], var))
            f.write('\n')

    if type(const_q) is str:
        const_q = None
    site = sd.Sim1D(site, working_directory=wd, run_dir=rd)

    site.rect_space_search(args[3], args[4], args[5], args[6], args[7], save=True, log_sample=args[-1])

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- elapsed time: {0}  ---".format(datetime.timedelta(seconds=(time.time() - start_time))))
