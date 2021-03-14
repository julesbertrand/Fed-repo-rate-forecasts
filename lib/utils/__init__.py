import pandas as pd


def check_columns(data: pd.DataFrame, columns: list = None, excl_columns: list = None) -> list:
    """
    Check that columns is a list
    Filters excl columns and columns not in dataset
    Check it is not empty

    Raises
    ------
    TypeError
        If columns is not a list
    """
    if excl_columns is None:
        excl_columns = []
    if columns is None:
        columns = list(data.columns)

    if not isinstance(columns, list) or not isinstance(excl_columns, list):
        raise TypeError("Invalid input: columns and excl_columns must be lists")
    columns = [col for col in columns if (col in data.columns and col not in excl_columns)]
    if not columns:
        msg = "Invalid input: columns does not overlap with data.columns \
or you put too many names in excl_columns"
        raise ValueError(msg)
    return columns
