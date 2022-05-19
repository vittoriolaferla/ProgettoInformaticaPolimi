import os
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

def plot_anomalies_count_dense(anomalies_cout, title=""):
    buckets = defaultdict(int)
    for faults in anomalies_cout.keys():
        faults_int = int(faults)
        if faults_int <= 39:
            buckets[faults_int] += anomalies_cout[faults]
        else:
            tenth = faults_int // 10
            buckets[tenth * 10] += anomalies_cout[faults]
    plot_anomalies_count(buckets, title=title)

def plot_anomalies_count(anomalies_count, title=""):
    figure, axis = plt.subplots(figsize=(25, 8))
    x = list(range(1, len(anomalies_count) + 1))
    x = [1.5 * i for i in range(1, len(anomalies_count) + 1)]
    sorted_key = sorted(anomalies_count.keys(), key=lambda x: int(x))
    print(sorted_key)
    y = [anomalies_count[key] for key in sorted_key]
    bars = axis.bar(x, y, edgecolor="black", width=0.8)
    axis.set_xticks(x)
    #axis.set_xlim(0, max(x))
    #axis.xaxis.set_major_locator(MultipleLocator(2))
    axis.set_xticklabels([str(i) for i in sorted_key])
    total_sum = sum(y)
    texts = []
    for key in sorted_key:
        percentage = (anomalies_count[key] / total_sum) * 100.0
        if percentage < 1.0:
            texts.append("< 1%")
        else:
            texts.append("{:.2f}%".format(percentage))
    y_min, y_max = axis.get_ylim()
    step = ((y_max - y_min) / len(x)) * 0.6
    for i, key in enumerate(texts):
        axis.text((i + 1) * 1.5, y[i] + step, texts[i], horizontalalignment="center")
    if title != "":
        axis.set_title(title)
    axis.set_xlabel("# of anomalies")
    axis.set_ylabel("# of fault outputs having that number of anomalies")
    plt.tight_layout()
    plt.savefig(title + ".svg", dpi=1200)

if __name__ == "__main__":
    anomalies_count_file_names = os.listdir("./")
    anomalies_count_file_names = [acfn 
        for acfn in anomalies_count_file_names
        if "anomalies_count.json" in acfn and "convolution" not in acfn
    ]
    global_anomalies_count = defaultdict(int)
    for acfn in anomalies_count_file_names:
        print(acfn)
        with open(acfn, "rb") as json_file:
            local_anomalies_count = json.load(json_file)
            for anomaly, count in local_anomalies_count.items():
                global_anomalies_count[anomaly] += count
    plot_anomalies_count_dense(global_anomalies_count, "Batch Norm S1 Faults Count")
    anomalies_frequency = {}
    total = sum(global_anomalies_count.values())
    for anomalies, count in global_anomalies_count.items():
        anomalies_frequency[int(anomalies)] = [count, "{:.5f}".format(count / total)]
    with open("batch_norm_S1_anomalies_count.json", "w") as json_file:
        json.dump(anomalies_frequency, json_file)
    