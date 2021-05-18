from frp.config import API_KEYS_FILEPATH, API_REQUESTS_PARAMS_FILEPATH, GETTERS
from frp.data_retrieval.get_data import get_data_from_apis
from frp.utils.files import open_file


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
        # providers=["OECD"],
        providers=GETTERS,
        save_dirpath="data/raw/",
    )

    # na_df = data[["date"]].copy()
    # na_df["na_count"] = data.notna().sum(axis=1)
    # na_df["month"] = na_df.date.dt.month
    # fig = na_df.plot.scatter(backend="plotly", x="date", y="na_count")
    # fig_path = filepath + "/datapoints.png"
    # fig.write_image(fig_path)


if __name__ == "__main__":
    main()
