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
from parsers import parse_user_mod

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
    user_mod = '/data/share/Japan/SiteInfo/S_B/VLER10_blind/refine_model.mod'
    name = site + '_' + rect_uni + '_' + str(iterations) + '_run_0_'
    wd = '/data/share/Japan/SiteInfo/S_B/VLER10_blind/'
    # rd = '/data/share/Japan/SiteInfo/S_B/{0}_Vs_MC_subl_{1}_smooth-{2}/'.format(site, motion, konno_ohmachi)
    rd = '/data/share/Japan/SiteInfo/S_B/VLER10_blind/'
    # TODO: Revert changes after Valerio blind test_cor_vs_space.py

    if not os.path.isdir(rd):  # if not a directory, make it
        print('creating dir: {0}'.format(rd))
        os.mkdir(rd)

    if os.path.exists(rd + name + '.cfg'):
        j = 0
        while os.path.exists(rd + name + '.cfg'):
            j += 1  # if the file already exists, give it another name
            name = name.replace('run_0', 'run_{0}'.format(j))
        print('name changed to: {0}'.format(name))
    if rect_uni == 'uniform':
        sub_uniform_search(site, wd, rd, 50, 10, iterations, name, 0, (0, 25), None, False, False, None,
                           (1, 5, 9), motion, konno_ohmachi)
    elif rect_uni == 'refine':
        refine_search(site, wd, rd, 200, iterations, user_mod, name, 0, (0, 25), None, False, False, None,
                      True,
                      motion, None, (250, np.exp(1)))

    else:
        rect_space_search(site, wd, rd, 100, 1500, iterations, name, 0, 5, False, (0, 25), None, False, False, None,
                          True, False, motion, None, (250, np.exp(1)), 0.5, 1, True, 0.5)


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

    with open(rd + name + '.cfg', 'wt') as f:
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


def refine_search(*args):
    site, wd, rd, = args[0:3]
    name = args[6]
    titles = ['site', 'data_dir', 'write_dir', 'delta', 'iterations', 'refine_model_fname', 'name',
              'i_ang', 'x_range_for_xcor', 'const_q', 'elastic', 'cadet_correct',
              'fill_troughs', 'save', 'motion', 'konno_ohmachi_smoothing(b)', 'log_spacing, base']

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

    site.refined_rect_space(model=parse_user_mod(args[5]), delta=args[3], iterations=args[4], name=name, i_ang=args[7],
                            x_cor_range=args[8], const_q=const_q, elastic=args[10], cadet_correct=args[11],
                            fill_troughs_pct=args[12], save=True, motion=args[13], konno_ohmachi=None,
                            log_sample=None)  # Log sample will be args[15] when not looking at Velerio's model.


def rect_space_search(*args):
    site, wd, rd, = args[0:3]
    name = args[6]
    titles = ['site', 'data_dir', 'write_dir', 'low', 'high', 'iterations', 'name',
              'i_ang', 'spacing[m]', 'force_min_spacing', 'x_range_for_xcor', 'const_q', 'elastic', 'cadet_correct',
              'fill_troughs', 'save',
              'debug', 'motion', 'konno_ohmachi_smoothing(b)', 'log_spacing, base', 'cor_co', 'std_dv(ln units)',
              'repeat_layers', 'repeat_chance']

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

    site.rect_space_search(low=args[3], high=args[4], iterations=args[5], name=name, i_ang=args[7], spacing=args[8],
                           force_min_spacing=args[9], x_cor_range=args[10], const_q=args[11], elastic=args[12],
                           cadet_correct=args[13], fill_troughs_pct=args[14], save=args[15], debug=args[16],
                           motion=args[17], konno_ohmachi=args[18], log_sample=None, cor_co=args[20],
                           std_dv=args[21], repeat_layers=args[22], repeat_chance=args[23])


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- elapsed time: {0}  ---".format(datetime.timedelta(seconds=(time.time() - start_time))))