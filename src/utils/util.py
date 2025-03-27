import yaml
import sys
import os


def yaml_parser(yaml_path):
    with open(yaml_path, "r") as stream:
        try:
            config_loaded = yaml.safe_load(stream)
        except yaml.YAMLError:
            msg = "Error while loading the yaml file : {}".format(yaml_path)
            print(msg)
            sys.exit(1)
    return config_loaded


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def get_parent(path, levels=1):
    common = path
    for i in range(levels+1):
        common = os.path.dirname(common)
    return common


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
