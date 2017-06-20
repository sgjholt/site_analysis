import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import seaborn as sb

v_mod = np.array([100, 200, 600, 300, 900, 1700, 1700])
scale = 1  # ln units for std
cor_co = 0.5
count = 1000
dists = np.zeros((v_mod.size, count))

for i, vel in enumerate(v_mod):
    a, b = np.max(np.array([(np.log(30) - np.log(vel)) / scale, -2])), np.min(
        np.array([(np.log(5000) - np.log(vel)) / scale, 2]))
    pdf = stats.truncnorm(a, b, np.log(vel), scale=scale)
    dists[i] = (pdf.rvs(count) - np.log(vel)) / scale

# fig, ax = plt.subplots(2, 3)

# for i, dist in enumerate(dists):
#    plt.subplot(2, 3, i+1)
#    plt.hist(dist, bins=200)
#    plt.title('Layer ' + str(i+1))


for i, c_layer in enumerate(dists[1:]):
    all_replaced = False
    tmp = (cor_co * dists[i]) + c_layer * (np.sqrt(1 - cor_co ** 2))
    while not all_replaced:
        locs = np.where((tmp <= np.log(75)) & (tmp <= np.log(5000)))
        replace = (pdf.rvs(len(locs[0])) - np.log(v_mod[i + 1])) / scale
        tmp[locs] = (cor_co * dists[i][locs]) + c_layer[locs] * (np.sqrt(1 - cor_co ** 2))
    dists[i + 1] = tmp
    #  dists[i + 1] = (cor_co * dists[i]) + c_layer * (np.sqrt(1 - cor_co ** 2))


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
plt.ylim([0.5, 6.5])
plt.gca().invert_yaxis()




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
