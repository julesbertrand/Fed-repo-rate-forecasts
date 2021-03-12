import datetime as dt
from typing import Tuple, Union, List
from loguru import logger

import pandas as pd

from lib.utils.df_utils import merge_df_list_on


class MinimalGetter:
    def __init__(self):
        pass

    @staticmethod
    def _check_response_status_code(response):
        """check if any HTTPError or any warning was raised during the request"""
        response.raise_for_status()
        if response.status_code != 200:
            logger.warning(f"Request ended with status code {response.status_code}")

    def get_data(
        self, series_params: list, start_date: dt.date, end_date: dt.date = None
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """Method to fetch data from api, clean it,
        and give it back in a dataframe with a metadata dict
        """
        raise NotImplementedError

    def _fetch_data(self, **kwargs):
        raise NotImplementedError


class TemplateGetter(MinimalGetter):
    def __init__(self, api_key, api_endpoint, date_format):
        self.api_key = api_key
        self.url = api_endpoint
        self.date_format = date_format
        MinimalGetter.__init__(self)

    def _parse_date(self, date: dt.date):
        """Parse dates based on the date format of teh data provider"""
        parsed_date = date.strftime(self.date_format)
        return parsed_date

    def get_data(
        self, series_params: list, start_date: dt.date, end_date: dt.date = None
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """Method to fetch data from api, clean it, and return a dataframe and a metadata dict

        Parameters
        ----------
        series_params: list
            List of ids US BLS series e.g.
        start_date: datetime.date
            The start of the observation period.
        end_date: datetime.date
            The start of the observation period. If None, will take today as default end date.

        Returns
        -------
        obs_df: pd.DataFrame
            DataFrame including one date column and one value column for every series retrieved.
        metadata_list: list
            List of dictionaries with metadata for every series retrieved.
        """
        logger.info(f"Retrieving data at {self.url} ...")
        obs_data_list, metadata_list = self.get_multiple_series(
            series_params, start_date, end_date
        )
        obs_df = self.clean_received_data(obs_data_list, metadata_list)
        logger.info("Data retrieved and cleaned successfully.")
        return obs_df, metadata_list

    def get_multiple_series(
        self, series_params: list, start_date: dt.date, end_date: dt.date = None
    ) -> Tuple[list, list]:
        raise NotImplementedError

    def get_series(
        self, series_params: Union[dict, str], start_date: dt.date, end_date: dt.date = None
    ) -> dict:
        raise NotImplementedError

    def _get_series_metadata(self, **kwargs) -> dict:
        raise NotImplementedError

    def _series_cleaner(self, obs_data: Union[dict, list]) -> pd.DataFrame:
        raise NotImplementedError

    def clean_series(self, obs_data: Union[list, dict]) -> pd.DataFrame:
        """For each observation dict, apply a cleaner specific to the source data format

        Parameters
        ----------
        obs_data: Union[list, dict]
            list or dict of observations for one series

        Returns
        -------
        pd.DataFrame
            Dataframe with columns 'value' type float and 'date' in datetime format
        """
        if len(obs_data) > 0:
            cleaned_series = self._series_cleaner(obs_data=obs_data)
        else:  # no obs
            cleaned_series = pd.DataFrame(columns=["date", "value"])
        cleaned_series["date"] = pd.to_datetime(cleaned_series["date"], format=self.date_format)
        cleaned_series = cleaned_series.astype({"value": float})
        return cleaned_series

    def clean_received_data(self, obs_data_list: list, metadata_list: list) -> pd.DataFrame:
        """Clean output from self.get_multiple_series to get an aggregated dataframe

        Parameters
        ----------
        obs_data_list: list
            List of all fred responses bodies in json format
        metadata_list
            List of dicts containing metadata about every retrieved series

        Raises
        ------
        KeyError
            If 'series_id' not in info_data keys

        Returns
        -------
        pd.DataFrame
            Aggregated data with names, in the right format
        """
        cleaned_data_list = []
        for i, obs_data in enumerate(obs_data_list):
            if metadata_list[i].get("series_id") is None:
                raise KeyError("No 'series_id' in info_data: this key is mandatory.")
            cleaned_series = self.clean_series(obs_data)
            series_name = self._give_name_to_series(metadata_list[i])
            cleaned_series.rename(columns={"value": series_name}, inplace=True)
            cleaned_data_list.append(cleaned_series)

        merged_data = merge_df_list_on(cleaned_data_list, on="date")
        return merged_data

    @staticmethod
    def _give_name_to_series(series_info: dict) -> str:
        """Give name to series using metadata about the series"""
        name_components = []
        if "series_id" not in series_info.keys():
            raise KeyError("No 'series_id' in info_data: this key is mandatory.")
        fields_list = [
            "series_id",
            "frequency",
            "aggregation_method",
            "seasonal_adjustment",
        ]
        for field in fields_list:
            if series_info.get(field):
                name_components.append(series_info[field])
        series_name = "_".join(name_components).replace(" ", "_")
        return series_name
