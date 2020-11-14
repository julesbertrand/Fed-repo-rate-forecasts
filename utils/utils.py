import numpy as np
import pandas as pd
import datetime as dt
from glob import glob
import ntpath
import os
import joblib
import yaml


def get_file_names(path="Data/", extension=".csv"):
    """
    Retrieve all data files names in a directory
    input: path
    output: list of file names
    """
    temp = glob(path + "/*" + extension)
    file_names = []
    for file_name in temp:
        _, file_name = ntpath.split(file_name)
        file_names.append(file_name)
    return file_names


def open_file(path, sep=";"):
    _, extension = path.rsplit(".", 1)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if extension == "csv":
        f = pd.read_csv(path, sep=sep)
    elif extension == "yaml":
        with open(path, "r") as file_name:
            f = yaml.safe_load(file_name)
    else:
        f = joblib.load(path)
    return f


def open_files(path, file_names):
    if isinstance(file_names, str):
        return open_file(path, file_names)
    f_dict = {}
    for f_name in file_names:
        f_dict[f_name] = open_file(path, f_name)
    return f_dict


def save_file(path, file_name, data, replace=False):
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(path):
        raise FileNotFoundError
    file_name, extension = file_name.split(".")
    if replace:
        try:
            os.remove(file_name)
        except OSError:
            pass  # if replace and the file does not exist, don't care
    else:
        i = 0
        while True:
            temp_name = ".".join((file_name + "_{:d}".format(i), extension))
            if os.path.exists(path + temp_name):
                i += 1
            else:
                break
        file_name += "_{:d}".format(i)
    file_name = ".".join((file_name, extension))
    if extension == "csv":
        data.to_csv(
            path + file_name,
            index=False,
            sep=";",
            encoding="utf-8",
        )
    elif extension == "yaml":
        with open(path + file_name, "w") as file_name:
            yaml.dump(data, file_name)
    else:
        joblib.dump(data, path + file_name, compress=1)


def save_files(path, files, replace=False):
    """
    files is a dict file_name: file
    replace = True means that files sith same files_names as the provided ones will be overwritten, carefull
    """
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(path):
        os.makedirs(path)
    files_not_saved = []
    for file_name, file in files.items():
        try:
            save_file(path=path, data=file, file_name=file_name, replace=replace)
        except Exception as e:
            files_not_saved.append(file_name + ": " + str(e))
    print(
        " {:d} file(s) saved succesfully in {:s} ".format(
            len(files.keys()) - len(files_not_saved), path
        ).center(120, "-")
    )
    if len(files_not_saved) > 0:
        print(
            " {:d} file(s) could not be saved: ".format(len(files_not_saved)).center(
                120, "-"
            )
        )
        print("\n".join(files_not_saved))
