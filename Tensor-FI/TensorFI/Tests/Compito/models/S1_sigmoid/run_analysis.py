import os
import os.path
import experiments_analyzer

def get_paths():
    root_path = "/home/aleto/experiments_data/sigmoid_S1"
    file_names = os.listdir(root_path)
    file_names = [file_name for file_name in file_names if "mode" not in file_name]
    groups = {}
    paths = []
    for file_name in file_names:
        mode_igid = file_name[5:-4]
        if mode_igid not in groups:
            groups[mode_igid] = [file_name]
        else:
            groups[mode_igid].append(file_name)
    for exps in groups.values():
        paths += [os.path.join(root_path, file_name) for file_name in exps]
    return paths

if __name__ == "__main__":
    root_path = "/home/aleto/experiments_data/sigmoid_S1"
    file_names = os.listdir(root_path)
    file_names = [file_name for file_name in file_names if "mode" not in file_name]
    groups = {}
    for file_name in file_names:
        mode_igid = file_name[5:-4]
        if mode_igid not in groups:
            groups[mode_igid] = [file_name]
        else:
            groups[mode_igid].append(file_name)
    for exps in groups.values():
        paths = [os.path.join(root_path, file_name) for file_name in exps]
        experiments_analyzer.main_with_args(paths)