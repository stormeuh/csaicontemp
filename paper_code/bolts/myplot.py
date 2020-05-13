from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


boltz = "data\\mc_Q_no_shield_real_2.dat"
shield = "data\\mc_Q_shield_real_2.dat"

fig, ax = plt.subplots()  # Create a figure containing a single axes.
#ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.

boltz_data = []
with open(boltz, "r") as f:
    lines = [x.strip() for x in f]
    for l in lines:
        iteration, elapsedtime, score, cumreward, goal_reached, numactions, agent_optimal = l.split(",")
        boltz_data.append((iteration, elapsedtime, score, cumreward, goal_reached, numactions, agent_optimal))

shield_data = []
with open(shield, "r") as f:
    lines = [x.strip() for x in f]
    for l in lines:
        iteration, elapsedtime, score, cumreward, goal_reached, numactions, agent_optimal = l.split(",")
        shield_data.append((iteration, elapsedtime, score, cumreward, goal_reached, numactions, agent_optimal))


"""
xs0 = range(len(boltz_data))
ys0 = []

xs1 = [x[0] for x in shield_data]
ys1 = []

N = 100
cumsum, moving_aves = [0], []

for i, x in enumerate(boltz_data, 1):
    x = int(x[2])
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)
ys0 = moving_aves



N = 100
cumsum, moving_aves = [0], []

for i, x in enumerate(shield_data, 1):
    x = int(x[2])
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)
ys1 = moving_aves


ax.plot(xs0[99:12000],ys0[0:12000-99], color="red", label="bolts")
ax.plot(xs1[99:12000],ys1[0:12000-99], color="blue", label="shields")
ax.set_xlabel('# episodes')  # Add an x-label to the axes.
ax.set_ylabel('moving average score (n=100)')  # Add a y-label to the axes.
ax.set_title("Moving average score in Minecraft")  # Add a title to the axes.
ax.legend()  # Add a legend.
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
fig.show()
"""

"""
xs0 = range(len(boltz_data))
ys0 = []

N = 100
cumsum, moving_aves = [0], []

for i, x in enumerate(boltz_data, 1):
    x = int(x[3])
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)
ys0 = moving_aves

xs1 = range(len(shield_data))
ys1 = []

N = 100
cumsum, moving_aves = [0], []

for i, x in enumerate(shield_data, 1):
    x = int(x[3])
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)
ys1 = moving_aves

ax.plot(xs0[99:12000],ys0[0:12000-99], color="red", label="bolts")
ax.plot(xs1[99:12000],ys1[0:12000-99], color="blue", label="shields")
ax.set_xlabel('# episodes')  # Add an x-label to the axes.
ax.set_ylabel('moving average reward (n=100)')  # Add a y-label to the axes.
ax.set_title("Moving average reward in Minecraft")  # Add a title to the axes.
ax.legend()  # Add a legend.
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
fig.show()
"""