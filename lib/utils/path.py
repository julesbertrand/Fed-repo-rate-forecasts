from pathlib import Path
import os
import re
import unidecode


def get_projet_root() -> Path:
    """Get project root directory path

    Returns
    -------
    pathlib.Path
    """
    return Path(__file__).parents[2]


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


def normalize_string(s: str, **kwargs) -> str:
    """Convert text to normalized text
    - lower
    - remove punctuation
    - remove trailing /multiple whitespaces
    - replace accents with letters (to unicode)
    """
    s = str(s)
    if kwargs.get("lower") is not False:
        s = s.lower()
    if kwargs.get("strip") is not False:
        s = s.strip()
        s = re.sub(r" +", " ", s)
    if kwargs.get("keep_char") is not False:
        if isinstance(kwargs.get("keep_char"), str):
            s = re.sub(kwargs.get("keep_char"), " ", s)
        else:
            s = re.sub(r"(?u)[^-$\+%\w. ]", "", s)
    if kwargs.get("unicode") is not False:
        s = unidecode.unidecode(s)
    return s
