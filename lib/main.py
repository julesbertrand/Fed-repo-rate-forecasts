import datetime as dt

import pandas as pd
from loguru import logger

from lib.data_retrieval.get_data import get_data_from_apis
from lib.utils.files import open_file, create_dir_if_missing, save_yaml
from config.config import API_KEYS_FILEPATH, API_REQUESTS_PARAMS_FILEPATH


def main():
    api_keys = open_file(API_KEYS_FILEPATH)
    api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
    data_start_date = api_params.pop("data_start_date")
    data_end_date = api_params.pop("data_end_date")

    data, metadata = get_data_from_apis(
        api_keys=api_keys,
        api_params=api_params,
        data_start_date=data_start_date,
        data_end_date=data_end_date,
        providers=["FRED", "USBLS", "OECD"],
    )

    date = dt.date.today().strftime("%Y%m%d")
    filepath = f"data/raw/{date}"
    create_dir_if_missing(filepath)
    data_path = filepath + "/raw_data.csv"
    data.to_csv(data_path, sep=";", index=False, encoding="utf-8")
    logger.info(f"Saved data to {data_path}.")
    metadata_path = filepath + "/metadata.yaml"
    save_yaml(metadata, metadata_path)
    logger.info(f"Saved metadata to {metadata_path}.")

    na_df = data[["date"]].copy()
    na_df["na_count"] = data.notna().sum(axis=1)
    na_df["month"] = na_df.date.dt.month
    fig = na_df.plot.scatter(backend="plotly", x="date", y="na_count")
    fig_path = filepath + "/datapoints.png"
    fig.write_image(fig_path)


if __name__ == "__main__":
    main()
