import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random

v_mod = np.array([100, 200, 600, 300, 900, 1700, 1700])
scale = 1  # ln units for std
cor_co = 0.5
dists = []
count = 1000
for vel in v_mod:
    a, b = np.max(np.array([(np.log(30) - np.log(vel)) / scale, -2])), np.min(
        np.array([(np.log(5000) - np.log(vel)) / scale, 2]))
    pdf = stats.truncnorm(a, b, np.log(vel), scale=scale)
    dists.append((pdf.rvs(1000) - np.log(vel)) / scale)

# fig, ax = plt.subplots(2, 3)

# for i, dist in enumerate(dists):
#    plt.subplot(2, 3, i+1)
#    plt.hist(dist, bins=200)
#    plt.title('Layer ' + str(i+1))


for i, c_layer in enumerate(dists[1:]):
    if i == 0:
        dists[i + 1] = (cor_co * dists[0]) + c_layer * (np.sqrt(1 - cor_co ** 2))
    else:
        dists[i + 1] = (cor_co * dists[i]) + c_layer * (np.sqrt(1 - cor_co ** 2))

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

