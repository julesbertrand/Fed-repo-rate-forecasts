import os
from pathlib import Path

import joblib
import pandas as pd
import yaml

from frp.utils.path import normalize_string


def get_valid_filename(filename: str) -> str:
    """Convert text to valid filename.
    - replace spaces with underscore
    - replace accent with letter
    - lower
    - remove trailing whitespaces
    - remove special characters
    eg. "The cat is blue in été" -> "the_cat_is_blue_in_ete
    """
    s = normalize_string(filename).replace(" ", "_")
    if s in [None, "", "_", " "]:
        raise ValueError(f"{filename} cannot be converted to a valid file name")
    return s


def open_yaml(filepath: str):
    """Open yaml file"""
    with open(filepath, "r") as data:
        content = yaml.safe_load(data)
    return content


def save_yaml(data, filepath: str):
    """Dump yaml file"""
    with open(filepath, "w") as f:
        yaml.dump(data, f)


def open_file(filepath: str, sep: str = ";"):
    """
    Open the file given its complete path.
    pandas red_csv if csv, yaml if yaml, joblib otherwise
    """
    _, extension = filepath.rsplit(".", 1)
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    if extension == "csv":
        content = pd.read_csv(filepath, sep=sep)
    elif extension == "yaml":
        content = open_yaml(filepath)
    else:
        content = joblib.load(filepath)
    return content


def open_files(dirpath: str, files_list: list) -> dict:
    """
    Open the files given their common path and all files names to retrieve.
    pandas red_csv if csv, yaml if yaml, joblib otherwise

    Returns
    -------
    dict
        Dict of filename: file_content
    """
    dirpath = Path(dirpath)
    files_dict = {}
    for filename in files_list:
        files_dict[filename] = open_file(dirpath / filename)
    return files_dict


def save_file(filepath: str, data, overwrite: bool = False):
    raise NotImplementedError


def save_files(dirpath: str, file_dict: dict, overwrite: bool = False):
    """
    Save a bunch of files

    Parameters
    ----------
    dirpath: str
        Diretory in which the save will be saved
    file_dict: dict
    overwrite: bool
        Whether to overwrite already existing file with save name in same directory
    """
    dirpath = Path(dirpath)
    files_not_saved = []
    for filename, file_data in file_dict.items():
        try:
            save_file(filepath=dirpath / filename, data=file_data, overwrite=overwrite)
        # pylint: disable=broad-except
        except Exception as exception:
            files_not_saved.append(filename + ": " + str(exception))
    nb_success = len(file_dict.keys()) - len(files_not_saved)
    print(f" {nb_success} file(s) saved succesfully in {dirpath} ".center(120, "-"))
    if len(files_not_saved) > 0:
        print(f" {len(files_not_saved)} file(s) could not be saved: ".format().center(120, "-"))
        print("\n".join(files_not_saved))
