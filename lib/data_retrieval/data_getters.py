from collections import defaultdict
import datetime as dt
from typing import Tuple, Union, List
from math import ceil
import re
import json
from copy import deepcopy
from loguru import logger
import requests

import numpy as np
import pandas as pd

from config.config import API_ENDPOINTS
from lib.data_retrieval.getter_templates import TemplateGetter, MinimalGetter
from lib.utils.df_utils import merge_df_list_on


class FREDGetter(TemplateGetter):
    api_endpoint = API_ENDPOINTS["FRED"]
    metadata_url = api_endpoint + "/series?"
    obs_url = api_endpoint + "/series/observations?"
    date_format = "%Y-%m-%d"
    earliest_realtime_start = "1776-07-04"
    latest_realtime_end = "9999-12-31"
    nan_char = "."
    max_results_per_request = 1000

    def __init__(self, api_key: str):
        TemplateGetter.__init__(
            self, api_key=api_key, api_endpoint=self.api_endpoint, date_format=self.date_format
        )

    # pylint: disable=arguments-differ
    def _fetch_data(self, url: str, params: dict = None) -> dict:
        response = requests.get(url=url, params=params)
        self._check_response_status_code(response)
        response_json = response.json()
        return response_json

    def get_multiple_series(
        self, series_params: list, start_date: dt.date, end_date: dt.date = None
    ) -> Tuple[list, list]:
        """Get multiple series data and info from FRED api
        and group in two lists (one info, one data)

        Parameters
        ----------
        series_params: list
            List of ids US BLS series e.g.
        start_date: datetime.date
            The start of the observation period. USBLS only sends data for whole years
        end_date: datetime.date
            The start of the observation period. USBLS only sends data for whole years

        Returns
        -------
        obs_data_list: list
            List of dictionaries with data reponse from FRED api in json format or dict
        metadata_list: list
            List of dictionaries with metadata for every series retrieved
        """
        if not isinstance(series_params, list):
            raise TypeError("'series_params' must be a list.")
        if len(series_params) == 0:
            raise ValueError("'series_params' must contain at least one element.")
        if end_date is None:
            end_date = dt.date.today()
        series_params = deepcopy(series_params)

        format_params = {
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": self._parse_date(start_date),
            "observation_end": self._parse_date(end_date),
        }

        obs_data_list = []
        metadata_list = []
        for params in series_params:
            params.update(format_params)
            obs_data = self._fetch_data(url=self.obs_url, params=params)
            info_data = self._get_series_metadata(params=params)
            obs_data_list.append(obs_data)
            metadata_list.append(info_data)
        return obs_data_list, metadata_list

    def get_series(
        self, series_params: dict, start_date: dt.date, end_date: dt.date = None
    ) -> dict:
        if not isinstance(series_params, dict):
            raise TypeError("'series_params' must be a dict.")
        obs_list = self.get_multiple_series(
            series_params=[series_params], start_date=start_date, end_date=end_date
        )
        return obs_list[0]

    # pylint: disable=arguments-differ
    def _get_series_metadata(self, params: dict) -> dict:
        series_metadata = self._fetch_data(url=self.metadata_url, params=params)
        series_metadata = series_metadata["seriess"][0]
        info_dict = {
            "provider": "FRED",
            "name": series_metadata.pop("title"),
            "series_id": params["series_id"],
            "frequency": params.get("frequency"),
            "units": series_metadata.pop("units_short"),
            "aggregation_method": params.get("aggregation_method"),
            "seasonal_adjustment": series_metadata.pop("seasonal_adjustment_short").lower(),
            "start_date": series_metadata.pop("observation_start"),
            "end_date": series_metadata.pop("observation_end"),
            "other": series_metadata,
        }
        return info_dict

    def _series_cleaner(self, obs_data: Union[dict, list]) -> pd.DataFrame:
        """Clean a series json response from FRED api

        Parameters
        ----------
        obs_data: json or dict
            body of the respone of the api

        Returns
        -------
        pd.DataFrame
            df with columns 'date' and 'value' cleaned
        """
        series = pd.json_normalize(obs_data["observations"])
        series.drop(columns=["realtime_start", "realtime_end"], inplace=True)
        series["value"] = series["value"].replace(self.nan_char, np.nan)
        return series


class USBLSGetter(TemplateGetter):
    api_endpoint = API_ENDPOINTS["USBLS"]
    date_format = "%Y"
    max_results_per_request = 1000
    max_years_per_request = 20

    def __init__(self, api_key: str):
        TemplateGetter.__init__(
            self, api_key=api_key, api_endpoint=self.api_endpoint, date_format=self.date_format
        )

    def __split_requests_time_range(self, start_date: dt.date, end_date: dt.date) -> list:
        """Split timerange from start_date to end_date by periods of self.max_years_per_request
        to avoid to get an error from USBLS.

        Parameters
        ----------
        start_date: datetime.date
        end_date: datetime.date

        Returns
        -------
        list
            List of tuples od periods (start_date_i, end_date_i)
            with end_date_i - start_date_i <= self.max_years_per_request

        """
        start_year = int(self._parse_date(start_date))
        end_year = int(self._parse_date(end_date))
        periods = []
        while end_year - start_year >= self.max_years_per_request:
            periods.append((str(start_year), str(start_year + self.max_years_per_request - 1)))
            start_year += self.max_years_per_request
        periods.append((str(start_year), str(end_year)))
        return periods

    # pylint: disable=arguments-differ
    def _fetch_data(self, url, payload: dict, headers: dict) -> dict:
        """Send a post request to USBLS API

        Raises
        ------
        HTPPError

        Returns
        -------
        dict
        """
        payload = json.dumps(payload)
        response = requests.post(url=url, data=payload, headers=headers)
        self._check_response_status_code(response)
        response_json = response.json()
        if response_json["status"] == "REQUEST_FAILED":
            raise requests.exceptions.RequestsWarning(
                "Message from US BLS API:\n" + "\n".join(response_json["message"])
            )
        if len(response_json["message"]) > 0:
            logger.warning("\n" + "\n".join(response_json["message"]))
        return response_json

    def get_series(
        self, series_params: str, start_date: dt.date, end_date: dt.date = None
    ) -> dict:
        if not isinstance(series_params, str):
            raise TypeError("'series_params' must be a str.")
        obs_list = self.get_multiple_series(
            series_params=[series_params], start_date=start_date, end_date=end_date
        )
        return obs_list[0]

    def get_multiple_series(
        self, series_params: list, start_date: dt.date, end_date: dt.date = None
    ) -> Tuple[list, list]:
        """Get multiple series data and info from FRED api
        and group in two lists (one info, one data)

        Parameters
        ----------
        series_params: list
            List of ids US BLS series e.g.
        start_date: datetime.date
            The start of the observation period. USBLS only sends data for whole years
        end_date: datetime.date
            The start of the observation period. USBLS only sends data for whole years

        Returns
        -------
        info_list: list
            List of dictionaries with metadata for every series retrieved
        data_list: list
            List of dictionaries with data reponse from FRED api in json format or dict
        """
        if not isinstance(series_params, list):
            raise TypeError("'series_params' must be a list.")
        if len(series_params) == 0:
            raise ValueError("'series_params' must contain at least one element.")
        if end_date is None:
            end_date = dt.date.today()

        periods = self.__split_requests_time_range(start_date, end_date)
        headers = {"Content-type": "application/json"}
        payload = {"registrationkey": self.api_key, "catalog": True, "seriesid": series_params}

        obs_dict = defaultdict(list)
        for start_year, end_year in periods:
            payload.update({"startyear": start_year, "endyear": end_year})
            response = self._fetch_data(url=self.api_endpoint, payload=payload, headers=headers)
            data = response["Results"]["series"]
            for i in range(len(series_params)):
                obs_dict[data[i]["seriesID"]] += data[i]["data"]
        obs_data_list = [obs_dict[id] for id in series_params]

        metadata_list = []
        for i, series_id in enumerate(series_params):
            catalog = data[i]["catalog"]
            info_dict = self._get_series_metadata(catalog, series_id)
            metadata_list.append(info_dict)

        return obs_data_list, metadata_list

    # pylint: disable=arguments-differ
    @staticmethod
    def _get_series_metadata(catalog: dict, series_id: str) -> dict:
        info_dict = {
            "provider": "USBLS",
            "name": catalog.pop("survey_name"),
            "series_id": series_id,
            "frequency": None,
            "units": None,
            "aggregation_method": None,
            "seasonal_adjustment": re.sub(r"[^NSA]", r"", catalog.pop("seasonality")).lower(),
            "start_date": "To be Implemented",
            "end_date_date": "To be Implemented",
            "other": catalog,
        }
        return info_dict

    def _series_cleaner(self, obs_data: Union[dict, list]) -> pd.DataFrame:
        """Clean a series json response from USBLS api

        Parameters
        ----------
        obs_data: json or dict
            body of the respone of the api

        Returns
        -------
        pd.DataFrame
            df with columns 'date' and 'value' cleaned
        """
        series = pd.DataFrame(obs_data)
        series["date"] = pd.to_datetime(
            series["year"] + "-" + series["periodName"] + "-01", format="%Y-%B-%d"
        )
        series.drop(columns=series.columns.difference(["value", "date"]), inplace=True)
        return series


class OECDGetter(MinimalGetter):
    api_endpoint = API_ENDPOINTS["OECD"]
    date_format = "{:4d}-Q{:1d}"
    max_results_per_request = 1e6
    max_url_length = 1e3

    def __init__(self, api_key: str):
        self.api_key = api_key
        MinimalGetter.__init__(self)

    def __build_oecd_request_url(self, dataset_id: str, dimensions: list):
        dimensions_agg = ["+".join(d) for d in dimensions]
        dimensions_agg = ".".join(dimensions_agg)
        url = f"{self.api_endpoint}/{dataset_id}/{dimensions_agg}/all"
        return url

    @staticmethod
    def _datetime_to_oecd_date_format(date: dt.date) -> str:
        year, month = map(int, date.strftime("%Y-%m").split("-"))
        quarter = ceil(month / 3)
        new_date = f"{year}-Q{quarter}"
        return new_date

    @staticmethod
    def _oecd_date_to_datetime_format(date: str) -> dt.date:
        year, quarter = int(date[:4]), int(date[-1])
        new_date = dt.date(year=year, month=quarter * 3)
        return new_date

    # pylint: disable=arguments-differ
    def _fetch_data(self, url: str, params: dict = None) -> Union[dict, list]:
        response = requests.get(url=url, params=params)
        self._check_response_status_code(response)
        response_json = response.json()
        return response_json

    def get_data(
        self, series_params: list, start_date: dt.date, end_date: dt.date = None
    ) -> Tuple[pd.DataFrame, List[dict]]:
        if not isinstance(series_params, list):
            raise TypeError("'series_params' must be a list.")
        if len(series_params) == 0:
            raise ValueError("'series_params' must contain at least one element.")
        if end_date is None:
            end_date = dt.date.today()

        metadata_list = []
        obs_df_list = []
        for dataset_series_params in series_params:
            obs_df, metadata_sublist = self.get_data_from_one_dataset(
                series_params=dataset_series_params, start_date=start_date, end_date=end_date
            )
            metadata_list += metadata_sublist
            obs_df_list.append(obs_df)

        merged_data = merge_df_list_on(obs_df_list, on="date")
        return merged_data, metadata_list

    def get_data_from_one_dataset(
        self, series_params: dict, start_date: dt.date, end_date: dt.date = None
    ) -> Tuple[pd.DataFrame, List[dict]]:
        if not isinstance(series_params, dict):
            raise TypeError("'series_params' must be a dict.")
        if end_date is None:
            end_date = dt.date.today()
        series_params = deepcopy(series_params)

        dataset_id = series_params.pop("dataset_id")
        dimensions = list(series_params.get("dimensions").values())
        url = self.__build_oecd_request_url(dataset_id, dimensions)

        params = {
            "startTime": self._datetime_to_oecd_date_format(start_date),
            "endTime": self._datetime_to_oecd_date_format(end_date),
            "dimensionAtObservation": "allDimensions",
            "pid": self.api_key,
        }

        logger.info(f"Retrieving data at {self.api_endpoint} in dataset '{dataset_id}'...")
        response = self._fetch_data(url=url, params=params)
        obs_list = response.get("dataSets")[0].get("observations")

        if len(obs_list) == 0:
            logger.warning(
                f"No available records for \n dimensions: {dimensions} \n parameters: {params}."
            )
            obs_df = []
            metadata_list = []
            return obs_df, metadata_list

        retrieved_dimensions = response.get("structure").get("dimensions").get("observation")
        retrieved_attributes = response.get("structure").get("attributes").get("observation")

        obs_df = pd.DataFrame(obs_list).transpose()
        obs_df = self._assign_attributes(obs_df=obs_df, attributes=retrieved_attributes)
        obs_df = self._assign_dimensions(obs_df=obs_df, dimensions=retrieved_dimensions)
        obs_df["date"] = pd.PeriodIndex(obs_df["period"], freq="Q").to_timestamp()

        temp_metadata = obs_df.drop(columns=["period", "date", "value"]).drop_duplicates()
        temp_metadata_list = temp_metadata.to_dict(orient="records")
        metadata_list = []
        for series_info in temp_metadata_list:
            info_dict = self._get_metadata(series_info)
            metadata_list.append(info_dict)

        data = obs_df.pivot_table(
            index=["date"], columns=["subject", "country", "measure"], values=["value"]
        )
        # remove unusefull level "values" from index
        data.columns = data.columns.droplevel(0)
        # flatten multindex to get custom series id
        data.columns = ["_".join(col).strip() for col in data.columns.values]
        # "date" moved from index to column
        data = data.reset_index(drop=False)

        logger.info("Data retrieved and cleaned successfully.")
        return data, metadata_list

    def _assign_attributes(self, obs_df: pd.DataFrame, attributes: list) -> pd.DataFrame:
        attr_col_names = ["value"] + list(map(lambda x: x.get("id").lower(), attributes))
        obs_df = obs_df.rename(columns=dict(enumerate(attr_col_names)))
        for i, col_name in enumerate(attr_col_names[1:]):
            attributes_values = attributes[i].get("values")
            if attributes_values is None or len(attributes_values) == 0:
                pass
            else:
                obs_df[col_name] = obs_df[col_name].apply(
                    lambda x: self.__extract_attribute_value(
                        attributes_values=attributes_values, key=x, id_or_name="id"
                    )
                )

        return obs_df

    def _assign_dimensions(self, obs_df: pd.DataFrame, dimensions: list) -> pd.DataFrame:
        obs_df["dimensions_ids"] = obs_df.index.map(lambda x: re.findall(r"\d+", x))
        for i, dim in enumerate(dimensions):
            dim_name = dim.get("name").lower()
            values = dim.get("values")
            key_pos = dim.get("keyPosition")
            if key_pos is None:
                key_pos = i

            obs_df[dim_name] = obs_df["dimensions_ids"].apply(
                lambda x: self.__extract_dimension_value(
                    dimensions_ids=x, dimensions_values=values, key_pos=key_pos, id_or_name="id"
                )
            )

        obs_df = obs_df.drop(columns=["dimensions_ids"]).reset_index(drop=True)
        return obs_df

    @staticmethod
    def __extract_dimension_value(
        dimensions_ids: list, dimensions_values: List[dict], key_pos: int, id_or_name: str = "id"
    ) -> str:
        if id_or_name not in ["id", "name"]:
            raise ValueError(
                f"Dimensions of index can only be 'id' or 'name'. Current value: {id_or_name}"
            )
        idx = dimensions_ids[key_pos]
        dim = dimensions_values[int(idx)].get("id")
        return dim

    @staticmethod
    def __extract_attribute_value(
        attributes_values: list, key: Union[int, None], id_or_name
    ) -> str:
        if id_or_name not in ["id", "name"]:
            raise ValueError(
                f"Dimensions of index can only be 'id' or 'name'. Current value: {id_or_name}"
            )
        if pd.isna(key):
            return np.nan
        attr = attributes_values[int(key)].get(id_or_name)
        return attr

    @staticmethod
    def _get_metadata(series_info: dict) -> dict:
        series_id = "_".join(
            [series_info.get("subject"), series_info.get("country"), series_info.get("measure")]
        )
        if re.search(r"SA$", series_info.get("measure")):
            seasonal_adjustment = "sa"
        else:
            seasonal_adjustment = "nsa"

        info_dict = {
            "provider": "OECD",
            "name": "To be implemented",
            "series_id": series_id,
            "frequency": series_info.get("frequency").lower(),
            "units": series_info.get("unit").lower(),
            "aggregation_method": None,
            "seasonal_adjustment": seasonal_adjustment,
            "start_date": "To be Implemented",
            "end_date_date": "To be Implemented",
            "other": series_info,
        }
        return info_dict
