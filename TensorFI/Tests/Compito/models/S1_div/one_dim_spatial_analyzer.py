import os
import os.path
import json
import numpy as np
from collections import OrderedDict, defaultdict

THRESHOLD = 0.05

BATCH = 1
CHANNELS = 256
HEIGHT = 13
WIDTH = 13
BLOCK_SIZE = 16

SAME_FEATURE_MAP_SAME_ROW = 0
SAME_FEATURE_MAP_SAME_COLUMN = 1
SAME_FEATURE_MAP_BLOCK = 2
SAME_FEATURE_MAP_RANDOM = 3
MULTIPLE_FEATURE_MAPS_BULLET_WAKE = 4
MULTIPLE_FEATURE_MAPS_BLOCK = 5
MULTIPLE_FEATURE_MAPS_SHATTER_GLASS = 6
MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS = 7
MULTIPLE_FEATURE_MAPS_UNCATEGORIZED = 8

def linear_index(b, c, h, w):
    return  (b * (CHANNELS * HEIGHT * WIDTH) +
            c * (HEIGHT * WIDTH) +
            h * (WIDTH) + w)

def extract_and_merge_indexes(spartial_data, dimension):
    indexes = []
    for spatial_batch in spartial_data:
        dimension_key = str(dimension)
        if dimension_key not in spatial_batch:
            continue
        indexes += spatial_batch[dimension_key]
    return indexes

def one_anomaly_analysis(spatial_data):
    buckets = defaultdict(int)
    indexes = extract_and_merge_indexes(spatial_data, 1)
    for index in indexes:
        l_index = index[0][1]
        buckets[l_index] += 1
    return buckets

def extract_strides(offsets):
    strides = np.zeros((offsets.shape[0] - 1), dtype=offsets.dtype)
    for i in range(1, offsets.shape[0]):
        strides[i - 1] = offsets[i] - offsets[i - 1]
    return np.sort(strides)

def extract_unique_channels(index):
    return np.unique(index[:, 1])

def pattern_to_tuple(pattern):
    pattern_list = []
    for key in sorted(pattern.keys()):
        pattern_list.append((key, tuple(pattern[key])))
    return tuple(pattern_list)

def init_dictionary_if_necessary(dictionary, key, default_value=0):
    if key not in dictionary:
        dictionary[key] = default_value

def increment_counter(dictionary, key):
    init_dictionary_if_necessary(dictionary, key)
    dictionary[key] += 1

def assign_if_greater(dictionary, key, value):
    init_dictionary_if_necessary(dictionary, key)
    if value > dictionary[key]:
        dictionary[key] = value

def get_common_points(indexes):
    frequencies = defaultdict(int)
    for i in range(indexes.shape[0]):
        point = (indexes[i, 2], indexes[i, 3])
        increment_counter(frequencies, point)
    max_frequency = max(frequencies.values())
    common_points = [key for key in frequencies.keys() if frequencies[key] == max_frequency]
    common_points.sort(key=lambda x: x[0] * WIDTH + x[1])
    return common_points


def multi_anomaly_analysis(spatial_data, dimension):
    buckets = {}
    max_offset = {}
    indexes = extract_and_merge_indexes(spatial_data, dimension)
    print("Dimension: {}.".format(dimension))
    classified = 0
    for index in indexes:
        np_index = np.array(index)
        #np_index = np_index[:, 1]
        #unique_channels = extract_unique_channels(np_index)
        l_indexes = np_index[:, 1]#np.array([linear_index(*np_index[i, :]) for i in range(np_index.shape[0])])
        starting_l_index = l_indexes.min()
        offsets = np.sort(l_indexes - starting_l_index)
        strides = extract_strides(offsets)
        if np.unique(strides).size == 1:
            #CASO REGOLARE
            if strides[0] % BLOCK_SIZE == 0:
                init_dictionary_if_necessary(buckets, SAME_FEATURE_MAP_BLOCK, default_value=defaultdict(int))
                buckets[SAME_FEATURE_MAP_BLOCK][tuple(offsets // BLOCK_SIZE) ] += 1
                classified += 1
            else:
                init_dictionary_if_necessary(buckets, SAME_FEATURE_MAP_SAME_ROW, default_value=defaultdict(int))
                buckets[SAME_FEATURE_MAP_SAME_ROW][tuple(offsets)] += 1
                classified += 1
        elif np.unique(strides).count() == 2:
            init_dictionary_if_necessary(buckets, MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS, default_value=defaultdict(int))
            buckets[MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS][tuple(offsets)] += 1
        else:
            init_dictionary_if_necessary(buckets, SAME_FEATURE_MAP_RANDOM, default_value=defaultdict(int))
            buckets[SAME_FEATURE_MAP_RANDOM][tuple(offsets)] += 1
        #if len(unique_channels) == 1:
        #    #SAME FEATURE MAP
        #    assert(len(strides) == dimension - 1)
        #    if (strides < WIDTH).all():
        #        #SAME FEATURE MAP, SAME ROW
        #        #MAX OFFSET ==> LINEAR OFFSET
        #        #increment_counter(buckets, SAME_FEATURE_MAP_SAME_ROW)
        #        init_dictionary_if_necessary(buckets, SAME_FEATURE_MAP_SAME_ROW, default_value=defaultdict(int))
        #        buckets[SAME_FEATURE_MAP_SAME_ROW][tuple(offsets)] += 1
        #        assign_if_greater(max_offset, SAME_FEATURE_MAP_SAME_ROW, offsets[-1])
        #    elif (strides == WIDTH).all():
        #        #SAME FEATURE MAP, SAME COLUMN
        #        #MAX OFFSET ==> COLUMN OFFSET
        #        #increment_counter(buckets, SAME_FEATURE_MAP_SAME_COLUMN)
        #        init_dictionary_if_necessary(buckets, SAME_FEATURE_MAP_SAME_COLUMN, default_value=defaultdict(int))
        #        buckets[SAME_FEATURE_MAP_SAME_COLUMN][tuple(offsets // WIDTH)] += 1
        #        assign_if_greater(max_offset, SAME_FEATURE_MAP_SAME_COLUMN, offsets[-1] // WIDTH)
        #    elif (strides % BLOCK_SIZE == 0).all():
        #        #SAME FEATURE MAP, DIFFERENT BLOCK
        #        #MAX OFFSET ==> BLOCK OFFSET
        #        #increment_counter(buckets, SAME_FEATURE_MAP_BLOCK)
        #        init_dictionary_if_necessary(buckets, SAME_FEATURE_MAP_BLOCK, default_value=defaultdict(int))
        #        buckets[SAME_FEATURE_MAP_BLOCK][tuple(offsets // BLOCK_SIZE)] += 1
        #        assign_if_greater(max_offset, SAME_FEATURE_MAP_BLOCK, offsets[-1] // BLOCK_SIZE)
        #    else:
        #        #SAME FEATURE MAP BUT DIFFERENT COLUMN, ROW AND BLOCK
        #        #MAX OFFSET ==> LINEAR
        #        init_dictionary_if_necessary(buckets, SAME_FEATURE_MAP_RANDOM, default_value=defaultdict(int))
        #        buckets[SAME_FEATURE_MAP_RANDOM][tuple(offsets)] += 1
        #        assign_if_greater(max_offset, SAME_FEATURE_MAP_RANDOM, offsets[-1])
        #    classified += 1
        #else:
        #    #DIFFERENT FEATURE MAPS
        #    if (strides % (HEIGHT * WIDTH) == 0).all():
        #        #BULLET WAKE
        #        #MAX OFFSET ==> FEATURE MAP
        #        init_dictionary_if_necessary(buckets, MULTIPLE_FEATURE_MAPS_BULLET_WAKE, default_value=defaultdict(int))
        #        buckets[MULTIPLE_FEATURE_MAPS_BULLET_WAKE][tuple(offsets // (HEIGHT * WIDTH))] += 1
        #        assign_if_greater(max_offset, MULTIPLE_FEATURE_MAPS_BULLET_WAKE, offsets[-1] // (HEIGHT * WIDTH))
        #        classified += 1
        #    elif (strides % BLOCK_SIZE == 0).all():
        #        #BLOCK SIZE DIFF
        #        #MAX OFFSET ==> BLOCK
        #        init_dictionary_if_necessary(buckets, MULTIPLE_FEATURE_MAPS_BLOCK, default_value=defaultdict(int))
        #        buckets[MULTIPLE_FEATURE_MAPS_BLOCK][tuple(offsets // BLOCK_SIZE)] += 1
        #        assign_if_greater(max_offset, MULTIPLE_FEATURE_MAPS_BLOCK, offsets[-1] // BLOCK_SIZE)
        #        classified += 1
        #    else:
        #        if dimension > 2:
        #            common_points = get_common_points(np_index)
        #            if len(common_points) == np_index.shape[0]:
        #                #CASE IN WHICH ALL THE ERRORS ARE SPREADED ACROSS DIFFERENT FEATURE MAPS
        #                init_dictionary_if_necessary(buckets, MULTIPLE_FEATURE_MAPS_UNCATEGORIZED, default_value=defaultdict(int))
        #                buckets[MULTIPLE_FEATURE_MAPS_UNCATEGORIZED][tuple(offsets)] += 1
        #                classified += 1
        #            else:
        #                cy, cx = common_points[0]
        #                common_feature_maps = 0
        #                cfms = []
        #                for i in range(np_index.shape[0]):
        #                    if (np_index[i, 2:] == (cy, cx)).all():
        #                        cfms.append(np_index[i, 1])
        #                        common_feature_maps += 1
        #                if 2 <= common_feature_maps <= len(unique_channels):
        #                    #CASE SHATTER GLASS
        #                    min_feature_map = cfms[0]#np_index[:, 1].min()
        #                    pattern = {}
        #                    for i in range(np_index.shape[0]):
        #                        feature_map = np_index[i, 1]
        #                        l_index = linear_index(*np_index[i, :])
        #                        common_point_linear_index = linear_index(0, feature_map, cy, cx)
        #                        init_dictionary_if_necessary(pattern, feature_map - min_feature_map, default_value=[])
        #                        pattern[feature_map - min_feature_map].append(l_index - common_point_linear_index)
        #                    if common_feature_maps == len(unique_channels):
        #                        init_dictionary_if_necessary(buckets, MULTIPLE_FEATURE_MAPS_SHATTER_GLASS, default_value=defaultdict(int))
        #                        buckets[MULTIPLE_FEATURE_MAPS_SHATTER_GLASS][pattern_to_tuple(pattern)] += 1
        #                    else:
        #                        init_dictionary_if_necessary(buckets, MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS, default_value=defaultdict(int))
        #                        buckets[MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS][pattern_to_tuple(pattern)] += 1
        #                    classified += 1
        #                else:
        #                    #UNCATEGORIZED CASE
        #                    init_dictionary_if_necessary(buckets, MULTIPLE_FEATURE_MAPS_UNCATEGORIZED, default_value=defaultdict(int))
        #                    buckets[MULTIPLE_FEATURE_MAPS_UNCATEGORIZED][tuple(offsets)] += 1
        #                    classified += 1
        #        else:  
        #            #CASE IN WHICH TWO ERRORS ARE ACROSS TWO FEATURE MAPS BUT WITHOUT ANY RELATIONSHIP.
        #            init_dictionary_if_necessary(buckets, MULTIPLE_FEATURE_MAPS_UNCATEGORIZED, default_value=defaultdict(int))
        #            buckets[MULTIPLE_FEATURE_MAPS_UNCATEGORIZED][tuple(offsets)] += 1
        #            classified += 1
    print(classified / len(indexes))
    return buckets, max_offset

def analyze_one_buckets(buckets):
    totals = sum(buckets.values())
    max_offset = [0, 0, 0, 0]
    frequencies = {}
    for key_index, count in buckets.items():
        percentage = count / totals
        if percentage >= THRESHOLD:
            frequencies[key_index] = percentage
        #else:
        #    if dimension > 1:
        #        pattern = json.loads(key_index)
        #        for dim in pattern.keys():
        #            for offset, _ in pattern[dim]:
        #                if int(offset) > max_offset[int(dim)]:
        #                    max_offset[int(dim)] = int(offset)
    return frequencies

def analyze_multi_buckets(buckets, max_offset):
    fault_type_frequencies = {}
    pattern_frequencies = {}
    count = sum([sum([y for y in x.values()]) for x in buckets.values()])
    for fault_type in buckets.keys():
        fault_type_frequencies[fault_type] = sum([y for y in buckets[fault_type].values()]) / count
    for fault_type in buckets.keys():
        fault_count = sum([y for y in buckets[fault_type].values()])
        for pattern in buckets[fault_type].keys():
            init_dictionary_if_necessary(pattern_frequencies, fault_type, default_value={})
            pattern_frequencies[fault_type][pattern] = buckets[fault_type][pattern] / fault_count
    revised_pattern_frequencies = {}
    for fault_type in pattern_frequencies.keys():
        max_offset = 0
        mmax_x = 0
        mmin_x = 0
        hmfp = 0
        init_dictionary_if_necessary(revised_pattern_frequencies, fault_type, default_value={})
        for pattern, frequency in pattern_frequencies[fault_type].items():
            if frequency >= THRESHOLD:
                revised_pattern_frequencies[fault_type][str(pattern)] = frequency
            else:
                max_offset = pattern[-1]
        random_probability = 1.0 - sum(revised_pattern_frequencies[fault_type].values())
        revised_pattern_frequencies[fault_type]["RANDOM"] = random_probability
        revised_pattern_frequencies[fault_type]["MAX"] = max_offset
    return fault_type_frequencies, revised_pattern_frequencies

def main():
    spatial_file_names = [sfn for sfn in os.listdir("./") if "spatial_data" in sfn]
    spatial_data = []
    dimensions = set()
    for sfn in spatial_file_names:
        with open(sfn, "rb") as json_file:
            spatial_data.append(json.load(json_file))
            for key in spatial_data[-1]:
                if int(key) not in dimensions:
                    dimensions.add(int(key))
    multi_frequencies_global = np.zeros((len(dimensions) - 1, 9))
    index = 0
    dimensions = sorted(dimensions)
    global_stats = {}
    for dimension in dimensions:
        if dimension == 1:
            buckets = one_anomaly_analysis(spatial_data)
            one_frequencies = analyze_one_buckets(buckets)
            if one_frequencies == {}:
                one_frequencies["RANDOM"] = 1.0
            global_stats[1] = one_frequencies
        else:
            buckets, max_offset = multi_anomaly_analysis(spatial_data, dimension)
            multi_frequencies, pattern_frequencies = analyze_multi_buckets(buckets, max_offset)
            global_stats[dimension] = {}
            global_stats[dimension]["FF"] = multi_frequencies
            global_stats[dimension]["PF"] = pattern_frequencies
            for fault_type, frequency in multi_frequencies.items():
                multi_frequencies_global[index, fault_type] = frequency
            index += 1
            np.savetxt("spatial_multi_frequencies.csv", multi_frequencies_global, fmt="%2.2f", delimiter=",")
            pass
    with open("pippo.json", "w") as json_file:
        json.dump(global_stats, json_file, default=str)

if __name__ == "__main__":
    main()