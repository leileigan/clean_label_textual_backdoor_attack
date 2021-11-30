import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Times New Roman Cyr')

x3 = [50, 100, 150, 200, 250, 300]
# x3 = [5, 6, 7, 8, 9, 10]
ag_y1 = [91.12, 89.92, 91.5, 91.6, 90.6, 90.5]
ag_y2 = [50.0, 76.67, 73.30, 76.67, 90.0, 90.0]

fig = plt.figure(figsize=(15, 7))

ax1 = fig.add_subplot()
ax2 = ax1.twinx()  # this is the important function
ax1.spines['bottom'].set_linewidth(4)
ax1.spines['left'].set_linewidth(4)
ax1.spines['top'].set_linewidth(4)
ax1.spines['right'].set_linewidth(4)
ax1.tick_params(labelsize=25)
ax2.tick_params(labelsize=25)

marker_size=15

ax1.plot(x3, ag_y2, color='g', marker='o', linestyle='-', linewidth='5', markersize=marker_size, label='AG\'s News ASR')
ax2.plot(x3, ag_y1, color='g', marker='x', linestyle='-.', linewidth='5', markersize=marker_size, label='AG\'s News CACC')

ax1.set_xticks([50, 100, 150, 200, 250, 300])
ax1.set_yticks([20, 40, 60, 80, 100])
ax1.set_xlabel("Poisoning Samples Number", fontsize=35)
ax1.set_ylabel('Attack Success Rate', fontsize=35)
#ax2.set_xlim([0, 550])
ax2.set_ylabel('Clean Accuracy', fontsize=35)
ax1.set_ylim([30, 105])
ax2.set_ylim([10, 105])

lines, labels = [], []
ax1_lines, ax1_labels = fig.axes[0].get_legend_handles_labels()
ax2_lines, ax2_labels = fig.axes[1].get_legend_handles_labels()
for line1, line2 in zip(ax1_lines, ax2_lines):
    lines.append(line1)
    lines.append(line2)
for label1, label2 in zip(ax1_labels, ax2_labels):
    labels.append(label1)
    labels.append(label2)

plt.legend(lines, labels, loc=(0.7, 0.03), fontsize=20)
plt.show()
plt.savefig('dev_ag.pdf', bbox_inches='tight', format='pdf')