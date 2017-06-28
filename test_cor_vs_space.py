import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import copy
import seaborn as sb

# v_mod = np.array([100, 200, 600, 300, 900, 1700, 1700])
v_mod = np.linspace(100, 2000, 21)
v_mod = np.append(v_mod, v_mod[-1])
scale = 1  # ln units for std
cor_co = 0.9
count = 100


def cor_v_space(v_mod, count, lower_v=75, upper_v=4000, cor_co=0.5, scale=1, plot=False, repeat_layers=False,
                repeat_chance=0.25):
    """

    :param v_mod:
    :param count:
    :param lower_v:
    :param upper_v:
    :param cor_co:
    :param scale:
    :param plot:
    :param repeat_layers:
    :param repeat_chance:
    :return:
    """
    dists = np.zeros((v_mod.size, count))

    for i, vel in enumerate(v_mod):
        a, b = np.max(np.array([(np.log(lower_v) - np.log(vel)) / scale, -2])), np.min(
            np.array([(np.log(upper_v) - np.log(vel)) / scale, 2]))
        pdf = stats.truncnorm(a, b, np.log(vel), scale=scale)
        dists[i] = (pdf.rvs(count) - np.log(vel)) / scale

    for i, c_layer in enumerate(dists[1:]):

        all_replaced = False

        if repeat_layers:
            tmp = copy.deepcopy(c_layer)
            lucky_draw = np.where(np.random.random(count) <= repeat_chance)
            tmp[lucky_draw] = dists[i][lucky_draw] + (np.log(v_mod[i]) - np.log(v_mod[i + 1]))
            inds = np.arange(0, count)
            others = np.array(list(set(inds) - set(lucky_draw[0])))
            print(others)
            if len(others) > 0:
                tmp[others] = (cor_co * dists[i][others]) + c_layer[others] * (np.sqrt(1 - cor_co ** 2))
            else:
                pass
        else:
            tmp = (cor_co * dists[i]) + c_layer * (np.sqrt(1 - cor_co ** 2))

        counter = 0
        while not all_replaced:
            counter += 1
            locs = np.where(((tmp * scale) + np.log(v_mod[i + 1]) <= np.log(lower_v)) | (
                (tmp * scale) + np.log(v_mod[i + 1]) > np.log(upper_v)))
            print(i, len(locs[0]), np.exp(tmp[locs] + np.log(v_mod[i + 1])))
            if len(locs[0]) == 0:
                all_replaced = True
            else:
                if counter < 1000:
                    replace = (pdf.rvs(len(locs[0])) - np.log(v_mod[i + 1])) / scale
                    tmp[locs] = (cor_co * dists[i][locs]) + replace * (np.sqrt(1 - cor_co ** 2))
                else:  # protects against getting stuck upper limit -
                    # usually only a problem if user wants high/unity correlation.
                    tmp[locs] = (np.log(upper_v) - np.log(v_mod[i + 1])) / scale

        dists[i + 1] = tmp
        #  dists[i + 1] = (cor_co * dists[i]) + c_layer * (np.sqrt(1 - cor_co ** 2))
    if plot:
        layers = [x + 0.5 for x in range(len(v_mod))]
        plt.figure()
        plt.step(v_mod, layers, 'k')
        counter = 0
        for model in np.exp((np.array(dists) * scale) + np.log(v_mod).reshape((len(v_mod)), 1)).T:
            plt.step(model, layers, 'r', linestyle='--')
            counter += 1
            if counter > count:
                break
        plt.xlabel('Velocity')
        plt.ylabel('Layer')
        plt.ylim([0.5, len(v_mod) + 0.5])
        plt.gca().invert_yaxis()

    else:
        return np.exp((np.array(dists) * scale) + np.log(v_mod).reshape((len(v_mod)), 1)).T


all_mods = []
for _ in range(1000):
    v_up, v_current = np.zeros(1), np.zeros(1)
    rand_mod = []
    for i, vels in enumerate(dists):
        if i == 0:  # pick value for top layer
            rand_mod.append(vels[random.randint(0, len(vels) - 1)])
            v_up = rand_mod[0]
        else:
            b = vels[random.randint(0, len(vels) - 1)]
            v_current = (cor_co * v_up) + b * (np.sqrt(1 - cor_co ** 2))
            rand_mod.append(v_current)
            v_up = v_current

    rand_mod = np.exp((np.array(rand_mod) * scale) + np.log(v_mod))
    all_mods.append(rand_mod)

layers = [x + 0.5 for x in range(len(rand_mod))]

plt.figure()
plt.step(v_mod, layers, 'k')
for rand_mod in all_mods:
    plt.step(rand_mod, layers, 'r', linestyle='--')
plt.xlabel('Velocity')
plt.ylabel('Layer')
plt.ylim([0.5, 6.5])
plt.gca().invert_yaxis()

# ---------------KAMAI | VS-Z | (2016) TESTING-------------------------------#
sb.set_style('white')
sb.set_context('talk')

Vs30s = [200, 300, 500, 700]
depths = np.linspace(0, 100, 101)

fig, axs = plt.subplots(1, len(Vs30s), sharey=True, sharex=True)
axs.flatten()[0].invert_yaxis()
for i, out in enumerate(zip(axs.flatten(), Vs30s)):
    ax, Vs30 = out
    ax.plot(kamai_vsz(depths, Vs30)[0], depths, 'k')
    ax.plot(kamai_vsz(depths, Vs30)[1][0], depths, 'r--')
    ax.plot(kamai_vsz(depths, Vs30)[1][1], depths, 'r--')
    if i == 0:
        ax.set_ylabel('$Depth$ $[m]$')
    ax.set_xlabel('$V_s$ $[m/s]$')
    ax.set_title('$V_s$$30$: ${0}$ $[m/s]$'.format(Vs30))
    ax.grid(which='both')

fig.suptitle('Kamai et al., 2016 | Vs-Z | Japan')
