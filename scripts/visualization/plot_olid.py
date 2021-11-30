import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Times New Roman Cyr')

x2 = [10, 20, 30, 40, 50]
#x2 = [1, 2, 3, 4, 5]
olid_y1 = [81.68, 81.97, 81.86, 81.7, 81.94]
olid_y2 = [88.0, 93.0, 96.0, 99.0, 99.0]

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

ax1.plot(x2, olid_y2, color='y', marker='o', linestyle='-', linewidth='5', markersize=marker_size, label='OLID ASR')
ax2.plot(x2, olid_y1, color='y', marker='x', linestyle='-.', linewidth='5', markersize=marker_size, label='OLID CACC')

ax1.set_xticks([10, 20, 30, 40, 50])
ax1.set_yticks([70, 80, 90, 100])
ax1.set_xlabel("Poisoning Samples Number", fontsize=35)
ax1.set_ylabel('Attack Success Rate', fontsize=35)
#ax2.set_xlim([0, 550])
ax2.set_ylabel('Clean Accuracy', fontsize=35)
ax1.set_ylim([70, 105])
ax2.set_ylim([10, 100])

lines, labels = [], []
ax1_lines, ax1_labels = fig.axes[0].get_legend_handles_labels()
ax2_lines, ax2_labels = fig.axes[1].get_legend_handles_labels()
for line1, line2 in zip(ax1_lines, ax2_lines):
    lines.append(line1)
    lines.append(line2)
for label1, label2 in zip(ax1_labels, ax2_labels):
    labels.append(label1)
    labels.append(label2)

plt.legend(lines, labels, loc=(0.75, 0.03), fontsize=20)
plt.show()
plt.savefig('dev_olid.pdf', bbox_inches='tight', format='pdf')