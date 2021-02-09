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
    input: path to directory, extension of files to retrieve
    output: list of file names
    """
    temp = glob(path + "/*" + extension)
    file_names = []
    for file_name in temp:
        _, file_name = ntpath.split(file_name)
        file_names.append(file_name)
    return file_names


def open_file(filepath: str, sep: str = ";"):
    """
    Open the file given its complete path.
    pandas red_csv if csv, yaml if yaml, joblib otherwise
    """
    _, extension = filepath.rsplit(".", 1)
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    if extension == "csv":
        f = pd.read_csv(filepath, sep=sep)
    elif extension == "yaml":
        with open(filepath, "r") as data:
            f = yaml.safe_load(data)
    else:
        f = joblib.load(filepath)
    return f


def open_files(path: str, file_names: list) -> dict:
    """
    Open the files given their common path and all files names to retrieve.
    pandas red_csv if csv, yaml if yaml, joblib otherwise
    Ouput: dict file_name : file_object
    """
    if path[-1] != "/":
        path += "/"
    if isinstance(file_names, str):
        return open_file(path + file_names)
    f_dict = {}
    for f_name in file_names:
        f_dict[f_name] = open_file(path + f_name)
    return f_dict


def save_file(filepath: str, data, overwrite: bool = False):
    """
    Save file in directory given by path
    Input: path: directory where to save the file
            file_name: str, name to give to the new file
            data: object to save
            overwrite: boolean, whether to overwrite already existing file with save name in same directory
    If the path does not exist or the file could not be saved, it will tell you
    """
    path, file_name = os.path.split(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError("This path does not exist.")
    file_name, extension = file_name.split(".")
    if overwrite:
        try:
            os.remove(filepath)
        except OSError:
            pass  # if overwrite and the file does not exist, don't care
    else:
        i = 0
        while True:
            temp_name = ".".join((file_name + "_{:d}".format(i), extension))
            if os.path.exists(path + temp_name):
                i += 1
            else:
                break
        file_name += "_{:d}".format(i) * (i > 0)
    file_name = ".".join((file_name, extension))
    filepath = path + "/" + file_name
    if extension == "csv":
        data.to_csv(
            filepath,
            index=False,
            sep=";",
            encoding="utf-8",
        )
    elif extension == "yaml":
        with open(filepath, "w") as file_name:
            yaml.dump(data, file_name)
    else:
        joblib.dump(data, filepath, compress=1)


def save_files(path: str, files: dict, overwrite: bool = False):
    """
    Save a bunch of files
    Input: path: diretory in which the save will be saved
            files: dict file_name: file object
            overwrite: whether to overwrite already existing file with save name in same directory
    If the path does not exist, then the directory will be created and the files saved in it
    """
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(path):
        os.makedirs(path)
    files_not_saved = []
    for file_name, file in files.items():
        try:
            save_file(filepath=path + file_name, data=file, overwrite=overwrite)
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
