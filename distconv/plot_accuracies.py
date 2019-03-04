#!/usr/bin/env python3

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import os

# disable interactive plotting
#plt.ioff()

sns.set()

# coding: utf-8
def parse_output_file(output):
    training_top1 = []
    training_top5 = []
    validation_top1 = []
    validation_top5 = []
    test_top1 = np.nan
    test_top5 = np.nan
    for line in output:
        line = line.strip()
        if " (instance 0) training epoch" in line:
            if "top-5 categorical accuracy" in line:
                training_top5.append(float(line.split()[-1][:-1]))
            elif "categorical accuracy" in line:
                training_top1.append(float(line.split()[-1][:-1]))
        elif " (instance 0) validation" in line:
            if "top-5 categorical accuracy" in line:
                validation_top5.append(float(line.split()[-1][:-1]))
            elif "categorical accuracy" in line:
                validation_top1.append(float(line.split()[-1][:-1]))
        elif "test top-5 categorical accuracy" in line:
            test_top5 = float(line.split()[-1][:-1])
        elif "test categorical accuracy" in line:
            test_top1 = float(line.split()[-1][:-1])
    assert training_top1
    assert training_top5
    assert validation_top1
    assert validation_top5
    assert test_top1
    assert test_top5
    return (training_top1, training_top5, validation_top1, validation_top5, test_top1, test_top5)

accuracies = {}
for exp_id in os.listdir("."):
    if not (exp_id.startswith("201") and "_lbann_" in exp_id):
        continue
    out_file = os.path.join(exp_id, "output.txt")
    if not os.path.isfile(out_file):
        continue
    sys.stderr.write("Reading output file: %s\n" % out_file)
    x = parse_output_file(open(out_file))
    print("%s\n" % str(x))
    accuracies[exp_id] = x

titles = ["Training Categorical Accuracy",
          "Training Top-5 Categorical Accuracy",
          "Validation Categorical Accuracy",
          "Validation Top-5 Categorical Accuracy"]

pdf = PdfPages("accuracies.pdf")

#fig = plt.figure()

for i in range(4):
    if i == 0 or i == 2:
        fig = plt.figure(figsize=(10, 14))
    ax = plt.subplot(2,1,i%2+1)
    ax.set_title(titles[i])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Epoch")
    for out in accuracies:
        ax.plot(accuracies[out][i],'-o', label=out)
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    ax.legend()
    #if i == 3:
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #ax.legend(bbox_to_anchor=(1.04, 0), loc="lower left")

    if i == 1 or i == 3:
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)

#plt.show()

color_map = {}
top1 = []
top5 = []
exps = []
colors = []
for out in accuracies:
    exps.append(out)
    top1.append(accuracies[out][4])
    top5.append(accuracies[out][5])
    cat = out.split("_")[3:]
    if not cat:
        cat = "default"
    else:
        cat = "_".join(cat)
    color = color_map.setdefault(cat, "C" + str(len(color_map) + 1))
    colors.append(color_map[cat])
fig = plt.figure(figsize=(10, 12))
ax = plt.subplot(2,1,1)
ax.barh(range(len(top1)), top1, color=colors)
ax.set_yticks(range(len(exps)))
ax.set_yticklabels(exps)
ax.set_title("Test Accuracy")


ax = plt.subplot(2,1,2)
ax.barh(range(len(top1)), top5, color=colors)
ax.set_yticks(range(len(exps)))
ax.set_yticklabels(exps)
ax.set_title("Test Top-5 Accuracy")

plt.tight_layout()

pdf.savefig()
pdf.close()
