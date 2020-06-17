import numpy as np
import pandas as pd 
import datetime as dt
from glob import glob
import ntpath
import os
import joblib


def get_file_names(path = "Data/", extension=".csv"):
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

def open_files(path, file_names):
    def open_file(path, file_name):
        _, extension = file_name.split(".")
        if extension == 'csv':
            f = pd.read_csv(path + file_name, sep=';')
        else:
            f = joblib.load(path + file_name)
        return f
    if isinstance(file_names, str):
        return open_file(path, file_names)
    f_dict = {}
    for f_name in file_names:
        f_dict[f_name] = open_file(path, f_name)
    return f_dict

def save_files(path, files, replace=False):  # files is a dict file_name: file
    # lack replace or not
    if not os.path.exists(path):
        os.makedirs(path)
    def save_file(path, f, file_name, replace=False):
        file_name, extension = file_name.split(".")
        if replace:
            try:
                os.remove(file_name)
            except OSError: pass
        else:
            i = 0
            while os.path.exists(path + ".".join((file_name + '_{:d}'.format(i), extension))):
                i += 1
            file_name += '_{:d}'.format(i)
        if extension == 'csv':
            file.to_csv(path + ".".join((file_name, extension)), index=False, sep=';', encoding='utf-8')
        else:
            joblib.dump(file, path + ".".join((file_name, extension)), compress = 1)
    files_not_saved = []
    for file_name, file in files.items():
        # try:
        save_file(path, file, file_name, replace=replace)
        # except Exception as e:
            # files_not_saved.append(file_name + ": " + str(e))
    print(" {:d} file(s) saved succesfully in {:s} ".format(len(files.keys()) - len(files_not_saved), path).center(120, "-"))
    if len(files_not_saved) > 0:
        print(" {:d} file(s) could not be saved: ".format(len(files_not_saved)).center(120, "-"))
        print("\n".join(files_not_saved))
    return


    
