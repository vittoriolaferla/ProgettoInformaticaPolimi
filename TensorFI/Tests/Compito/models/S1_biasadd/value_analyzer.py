import numpy as np 
import os
import os.path
from collections import defaultdict
from experiments_analyzer import load_outputs
import multiprocessing
import itertools

def to_json(dictionary, file_name):
    with open(file_name, "w") as json_file:
        json.dump(dictionary, json_file, default=int)

def from_json(file_name):
    with open(file_name, "r") as json_file:
        return json.load(json_file)

def classify_diff(golden, fault):
    golden_uint = np.frombuffer(golden, dtype=np.uint32)[0]
    fault_uint = np.frombuffer(fault, dtype=np.uint32)[0]
    golden_sign = np.bitwise_and(0x80000000, golden_uint)
    golden_exponent = np.bitwise_and(0x7F800000, golden_uint)
    golden_mantissa = np.bitwise_and(0x007FFFFF, golden_uint)
    fault_sign = np.bitwise_and(0x80000000, fault_uint)
    fault_exponent = np.bitwise_and(0x7F800000, fault_uint)
    fault_mantissa = np.bitwise_and(0x007FFFFF, fault_uint)
    is_sign = np.bitwise_xor(golden_sign, fault_sign)
    is_exponent = np.bitwise_xor(golden_exponent, fault_exponent)
    is_mantissa = np.bitwise_xor(golden_mantissa, fault_mantissa)
    return is_sign.astype(np.bool), is_exponent.astype(np.bool), is_mantissa.astype(np.bool)

def count_faults(golden_output, fault_output):
    fault_output_copy = np.copy(fault_output)
    nan_count = 0
    nan_map = np.isnan(fault_output)
    if nan_map.any():
        nan_count = np.sum(nan_map)
        fault_output_copy = np.nan_to_num(fault_output, nan=1E100)
    zero_map = fault_output_copy == 0.0
    golden_zero_map = golden_output == 0.0
    zero_map = np.logical_xor(zero_map, golden_zero_map)
    exclude_map = np.logical_or(zero_map, nan_map)
    zeros_count = zero_map.sum()
    diff = golden_output - fault_output_copy
    diff_map = np.abs(diff) > 1E-3
    faults_count = np.sum(diff_map) - nan_count - zeros_count
    non_nan_faults_map = np.logical_and(diff_map, np.logical_not(exclude_map))
    faults_count = np.sum(diff_map) - nan_count - zeros_count
    values_diff = []
    bit_classes = {}
    if np.sum(non_nan_faults_map) > 0:
        indexes = np.vstack(np.where(non_nan_faults_map)).T
        for i in range(indexes.shape[0]):
            b, c, y, x = indexes[i, :]
            values_diff.append(diff[b, c, y, x])
            classification = classify_diff(golden_output[b, c, y, x], fault_output[b, c, y, x])
            if tuple(classification) not in bit_classes:
                bit_classes[tuple(classification)] = 0
            bit_classes[tuple(classification)] += 1
    return faults_count, nan_count, zeros_count, np.array(values_diff), bit_classes
    


def get_data_paths():
    root_path = "/home/aleto/experiments_data/biasadd_S1"
    file_names = os.listdir(root_path)
    file_names = [file_name for file_name in file_names if "mode" not in file_name]
    return [os.path.join(root_path, file_name) for file_name in file_names]

def analyze_simulation_outputs(path):
    golden_output, fault_outputs = load_outputs([path])
    golden_output = golden_output[0]
    fault_outputs = fault_outputs[0]
    global_faults_count = 0
    global_nans_count = 0
    global_zeros_count = 0
    global_diff = np.zeros((0))
    global_bit_classes = {}
    for fault_output in fault_outputs:
        faults_count, nans_count, zeros_count, diff, bit_classes = count_faults(golden_output, fault_output)
        global_faults_count += faults_count
        global_nans_count += nans_count
        global_zeros_count += zeros_count
        global_diff = np.concatenate((global_diff, diff))
        for bit_class_key, bit_class_count in bit_classes.items():
            if bit_class_key not in global_bit_classes:
                global_bit_classes[bit_class_key] = 0
            global_bit_classes[bit_class_key] += bit_class_count
    return global_faults_count, global_nans_count, global_zeros_count, global_diff, global_bit_classes

def main():
    paths = get_data_paths()
    pool = multiprocessing.Pool()
    global_faults_count = 0
    global_nans_count = 0
    global_zeros_count = 0
    global_diff = np.zeros((0))
    global_bit_classes = {}
    for results in pool.imap_unordered(analyze_simulation_outputs, paths):
        global_faults_count += results[0]
        global_nans_count += results[1]
        global_zeros_count += results[2]
        global_diff = np.concatenate((global_diff, results[3]))
        for bit_class_key, bit_class_count in results[4].items():
            if bit_class_key not in global_bit_classes:
                global_bit_classes[bit_class_key] = 0
            global_bit_classes[bit_class_key] += bit_class_count
    total_faults = global_faults_count + global_nans_count + global_zeros_count
    nans_percentage = round(global_nans_count / total_faults, 6)
    zeros_percentage = round(global_zeros_count / total_faults, 6)
    less_than_1 = global_diff <= 1.0
    more_than_minus_1 = -1 <= global_diff
    within_interval = np.logical_and(less_than_1, more_than_minus_1).sum()
    within_interval_percentage = round(within_interval / total_faults, 6)
    outside_interval_percentage = round(1.0 - (within_interval_percentage + nans_percentage), 6)
    print("There have been {} faults".format(total_faults))

    print("[-1, 1]: {}".format(within_interval_percentage))
    print("Others: {}".format(outside_interval_percentage))
    print("NaN: {}".format(nans_percentage))
    print("Zeros: {}".format(zeros_percentage))
    print("Valid: {}".format(round(1.0 - nans_percentage - zeros_percentage, 6)))
    for bit_class_key in list(itertools.product((False, True), repeat=3)):
        if bit_class_key not in global_bit_classes:
            print("{}: 0.0".format(bit_class_key))
        else:
            percentage = round(global_bit_classes[bit_class_key] / global_faults_count, 6)
            print("{}: {}".format(bit_class_key, percentage))

if __name__ == "__main__":
    main()