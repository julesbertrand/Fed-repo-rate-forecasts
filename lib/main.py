import pandas as pd

from lib.data_retrieval.get_data import get_data_from_apis
from lib.utils.files import open_file

from config.config import API_KEYS_FILEPATH, API_REQUESTS_PARAMS_FILEPATH, API_ENDPOINTS


def main():
    api_keys = open_file(API_KEYS_FILEPATH)
    api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
    data_start_date = api_params["data_start_date"]

    data = get_data_from_apis(
        api_keys=api_keys,
        api_params=api_params,
        data_start_date=data_start_date,
    )

    return data


if __name__ == "__main__":
    print(main())
