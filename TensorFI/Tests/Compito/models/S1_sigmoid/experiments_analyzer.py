import numpy as np
import io
import tarfile
import os
import os.path
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
from scipy.stats import norm
import json

warnings.filterwarnings('ignore')

def plot_diff(diff):
    plt.hist(np.round(diff, decimals=2), bins=20, density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    mu, std = norm.fit(diff)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()

def is_bitflip(original, fault):
    origibal_buffer = np.frombuffer(original, dtype=np.uint32)
    fault_buffer = np.frombuffer(fault, dtype=np.uint32)
    bitwise_xor = np.bitwise_xor(origibal_buffer, fault_buffer)[0]
    binn = bin(bitwise_xor)
    return binn.count("1") == 1, binn

def are_continous(indexes, maximum):
    previous = indexes[0]
    print(indexes)
    for i in range(1, len(indexes)):
        if (indexes[i] - indexes[i - 1]) % (maximum - 1) > 1:
            return False
    return True

def plot_anomalies_count(anomalies_count, title=""):
    figure, axis = plt.subplots(figsize=(15, 8))
    x = list(range(1, len(anomalies_count) + 1))
    sorted_key = sorted(anomalies_count.keys())
    y = [anomalies_count[key] for key in sorted_key]
    axis.bar(x, y, edgecolor="black")
    axis.set_xticks(x)
    axis.set_xticklabels([str(i) for i in sorted_key])
    total_sum = sum(y)
    texts = []
    for key in sorted_key:
        percentage = (anomalies_count[key] / total_sum) * 100.0
        if percentage < 1.0:
            texts.append("< 1%")
        else:
            texts.append("{:.2f}%".format(percentage))
    for i, key in enumerate(texts):
        axis.text(i + 1, y[i] + 5.0, texts[i], horizontalalignment="center")
    if title != "":
        axis.set_title(title)
    axis.set_xlabel("# of anomalies")
    axis.set_ylabel("# of fault outputs having that number of anomalies")
    plt.tight_layout()
    plt.show()

def count_anomalies(golden_outputs, fault_outputs):
    anomalies_count = defaultdict(int)
    for i in range(len(golden_outputs)):
        golden_output = golden_outputs[i]
        for j, fault_output in enumerate(fault_outputs[i]):
            if np.isnan(fault_output).any():
                np.nan_to_num(fault_output, copy=False, nan=100E-100)
            anomalies = np.sum(np.abs(golden_output - fault_output) > 1E-3)
            if anomalies == 0:
                continue
            anomalies_count[int(anomalies)] += 1
    return anomalies_count

def read_numpy_array(extracted_tar_file):
    array_file = io.BytesIO()
    array_file.write(extracted_tar_file.read())
    array_file.seek(0)
    return np.load(array_file, allow_pickle=True)

def read_tar_file(tar_file_path):
    fault_outputs = []
    golden_output = None
    with tarfile.open(tar_file_path) as tar_file:
        for member in tar_file.getmembers():
            extracted_file = tar_file.extractfile(member)
            if extracted_file is None:
                continue
            if "output" in member.name:
                golden_output = read_numpy_array(extracted_file)
                continue
            try:
                numpy_array = read_numpy_array(extracted_file)
                fault_outputs.append(numpy_array)
            except:
                print("Unable to read: {}, skipping.".format(member.name))
                continue
    return golden_output, fault_outputs

def load_outputs(paths):
    golden_outputs = []
    fault_outputs = []
    for path in paths:
        golden_output_exp, fault_outputs_exp = read_tar_file(path)
        if golden_output_exp is None:
            print("{} does not have the golden output.".format(path))
            sys.exit(0)
        if fault_outputs_exp is None or len(fault_outputs_exp) == 0:
            print("{} does not have the fault outputs.".format(path))
            sys.exit(0)
        golden_outputs.append(golden_output_exp)
        fault_outputs.append(fault_outputs_exp)
    return golden_outputs, fault_outputs

def dump_to_file(dictionary, path):
    with open(path, "w") as json_file:
        json.dump(dictionary, json_file)

#def load_outputs(root_path, experiment, mode_igid, space_instance, image_instances):
#    golden_output = []
#    fault_outputs = []
#    for image_instance in image_instances:
#        tar_file_path = os.path.join(root_path, experiment, "S{}I{}_{}.tar".format(space_instance, image_instance, mode_igid))
#        print(tar_file_path)
#        go, fo = read_tar_file(tar_file_path)
#        golden_output.append(go)
#        fault_outputs.append(fo)
#    return golden_output, fault_outputs

def spatial_extractor(golden_outputs, fault_outputs):
    spatial_data = defaultdict(list)
    for i in range(len(golden_outputs)):
        golden_output = golden_outputs[i]
        for fault_output in fault_outputs[i]:
            equality = np.abs(golden_output - fault_output) > 1E-3
            anomalies = equality.sum()
            if anomalies == 0:
                continue
            #assert anomalies > 0
            indexes = np.vstack(np.where(equality)).T
            spatial_data[int(anomalies)].append(indexes.tolist())
    return spatial_data

def main_with_args(paths):
    experiment_name = os.path.basename(paths[0])[:2]
    mode_igid = os.path.basename(paths[0])[5:-4]
    print("Running {} - {}".format(experiment_name, mode_igid))
    golden_outputs, fault_outputs = load_outputs(paths)
    print("Loaded {} golden outputs and {} fault outputs.".format(len(golden_outputs), sum([len(fo) for fo in fault_outputs])))
    anomalies_count = count_anomalies(golden_outputs, fault_outputs)
    dump_to_file(anomalies_count, "{}_{}_anomalies_count.json".format(experiment_name, mode_igid))
    print("Extracted anomalies count and dumped.")
    spatial_data = spatial_extractor(golden_outputs, fault_outputs)
    assert len(anomalies_count.keys()) == len(spatial_data.keys())
    dump_to_file(spatial_data, "{}_{}_spatial_data.json".format(experiment_name, mode_igid))

def main():
    root_path = "/home/aleto/experiments_data"
    experiment = "convolution_S1"
    mode = "IOV"
    IGID = "PR"
    space_instances = 1
    image_instances = [1, 2]
    golden_outputs, fault_outputs = load_outputs(root_path, experiment, mode + "_" + IGID, space_instances, image_instances)
    print(golden_outputs[0].shape)
    anomalies_count = count_anomalies(golden_outputs, fault_outputs)
    #plot_anomalies_count(anomalies_count, experiment + " {} - {}".format(mode, IGID))
    one_fault_stat_model = np.zeros((33, 32), dtype=np.int)
    one_fault_diff = []
    for fault_output in fault_outputs[0]:
        print(fault_output.shape)
        difference = golden_outputs[0] - fault_output
        equality_map = np.abs(difference) > 1E-3
        anomalies = np.sum(equality_map)
        if anomalies == 1:
            index = np.where(equality_map)
            bitflip, one_hot = is_bitflip(golden_outputs[0][index], fault_output[index])
            if not bitflip:
                row = one_hot[2:].count("1")
                columns = [i for i in range(len(one_hot[2:])) if one_hot[2 + i] == "1"]
                diff = golden_outputs[0][index] - fault_output[index]
                for c in columns:
                    one_fault_stat_model[row, c] += 1
                one_fault_diff.append(diff)
        else:
            anomalies_indexes = np.vstack(np.where(equality_map)).T
            if len(anomalies_indexes) == 0:
                continue
            print("Contiguous in 0: {}".format(are_continous(anomalies_indexes[:, 0], golden_outputs[0].shape[0])))
            print("Contiguous in 1: {}".format(are_continous(anomalies_indexes[:, 1], golden_outputs[0].shape[1])))
            print("Contiguous in 2: {}".format(are_continous(anomalies_indexes[:, 2], golden_outputs[0].shape[2])))
            print("Contiguous in 3: {}".format(are_continous(anomalies_indexes[:, 3], golden_outputs[0].shape[3])))
    


if __name__ == "__main__":
    main()