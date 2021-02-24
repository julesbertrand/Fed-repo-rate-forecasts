from pathlib import Path
import os
import re
import yaml
import joblib

import pandas as pd


def get_projet_root() -> Path:
    """Get project root directory path

    Returns
    -------
    pathlib.Path
    """
    return Path(__file__).parent.parent.parent


def create_dir_if_missing(filepath: str):
    """Create directory if it does not exist already"""
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def list_files(dirpath: str, pattern: str = "*.csv") -> list:
    """
    List files in a directory
    """
    file_names = list(Path(dirpath).glob(pattern))
    return file_names


def get_valid_filename(filename: str) -> str:
    """Convert text to valid filename.
    - replace spaces with underscore
    - replace accent with letter
    - lower
    - remove trailing whitespaces
    - remove special characters
    eg. "The cat is blue in été" -> "the_cat_is_blue_in_ete
    """
    new_s = str(filename).strip().replace(" ", "_").lower()
    new_s = re.sub(r"(?u)[^-\w.]", r"", new_s)
    if new_s in [None, "", "_", " "]:
        raise ValueError(f"{filename} cannot be converted to a valid file name")
    return new_s


def open_yaml(filepath: str):
    with open(filepath, "r") as data:
        content = yaml.safe_load(data)
    return content


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
    """
    Save file in directory given by path
    If the file could not be saved, it will tell you

    Parameters
    ----------
    filepath: str
        Path where to save the file
    data:
        Python object to save
    overwrite: bool
        Whether to overwrite already existing file with save name in same directory
    """
    filepath = Path(filepath)
    dirpath = filepath.parent
    create_dir_if_missing(dirpath)
    filename = filepath.stem
    extension = filepath.suffix

    if overwrite:
        if os.path.exists(filepath):
            os.remove(filepath)
    else:
        i = 0
        temp_name = ".".join([filename + f"_{i}", extension])
        while os.path.exists(dirpath / temp_name):
            i += 1
            temp_name = ".".join([filename + f"_{i}", extension])
        filename += "_{:d}".format(i) * (i > 0)
        filename = ".".join([filename, extension])
        filepath = dirpath / filename

    if extension == "csv":
        data.to_csv(
            filepath,
            index=False,
            sep=";",
            encoding="utf-8",
        )
    elif extension == "yaml":
        with open(filepath, "w") as filename:
            yaml.dump(data, filename)
    else:
        joblib.dump(data, filepath, compress=1)


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
        except Exception as exception:
            files_not_saved.append(filename + ": " + str(exception))
    nb_success = len(file_dict.keys()) - len(files_not_saved)
    print(f" {nb_success} file(s) saved succesfully in {dirpath} ".center(120, "-"))
    if len(files_not_saved) > 0:
        print(f" {len(files_not_saved)} file(s) could not be saved: ".format().center(120, "-"))
        print("\n".join(files_not_saved))
