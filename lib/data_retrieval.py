from functools import reduce
from collections import defaultdict
import re
import json
from loguru import logger
import requests
import pandas as pd
from config.config import API_ENDPOINTS


def get_fred_data(api_key: str, series_params: list, start_date: str = None, end_date: str = None):
    """Get multiple series data and info from fred api
    and group in two lists (one info, one data)

    Parameters
    ----------
    api_key: str
        Your api key to access fred api
        see https://fred.stlouisfed.org/docs/api/api_key.html
    series_params: list
        List of dicts with series_params, including at least 'series_id'
    start_date: str
        The start of the observation period. YYYY-MM-DD
        https://fred.stlouisfed.org/docs/api/fred/series_observations.html#observation_start
    end_date: str
        The start of the observation period. YYYY-MM-DD
        https://fred.stlouisfed.org/docs/api/fred/series_observations.html#observation_end


    Returns
    -------
    info_list: list
        List of dictionaries with metadata for every series retrieved
    obs_list: list
        List of dictionaries with data reponse from fred api in json format
    """
    format_params = {
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    info_list = []
    obs_list = []
    for params in series_params:
        params.update(format_params)

        info_resp = requests.get(API_ENDPOINTS["FRED_API_URL_SER"], params=params)
        info_resp.raise_for_status()
        info_series = info_resp.json()["seriess"][0]

        obs_resp = requests.get(
            API_ENDPOINTS["FRED_API_URL_OBS"],
            params=params,
        )
        obs_resp.raise_for_status()
        obs_content = obs_resp.json()

        info_dict = {
            "name": info_series.get("title"),
            "series_id": params["series_id"],
            "frequency": params.get("frequency"),
            "units": params.get("units"),
            "aggregation_method": params.get("aggregation_method"),
            "seasonal_adjustment": params.get("seasonal_adjustment"),
        }

        info_list.append(info_dict)
        obs_list.append(obs_content)
    return obs_list, info_list


def clean_fred_data(obs_data_list: list, info_data_list: list) -> pd.DataFrame:
    """Clean output from get_fred_data to get an aggregated dataframe

    Parameters
    ----------
    obs_data_list: list
        List of all fred responses bodies in json format
    info_data_list
        List of dicts containing metadata about every retrieved serie

    Returns
    -------
    pd.DataFrame
        Aggregated data with names, in the right format
    """
    cleaned_data_list = []
    for i, obs_data in enumerate(obs_data_list):
        cleaned_data = clean_fred_serie(obs_data=obs_data, info_data=info_data_list[i])
        cleaned_data_list.append(cleaned_data)

    merged_data = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        cleaned_data_list,
    )

    merged_data["date"] = pd.to_datetime(merged_data["date"], format="%Y-%m-%d")
    data_types = {c: float for c in merged_data.columns.difference(["date"])}
    merged_data = merged_data.astype(data_types)
    return merged_data


def clean_fred_serie(obs_data, info_data: dict) -> pd.DataFrame:
    """Clean a series json response from fred api

    Parameters
    ----------
    obs_data: json or dict
        body of the respone of the api
    info_data: dict
        Information on teh serie, including at least 'series_id'

    Raises
    ------
    KeyError
        If 'series_id' not in info_data keys

    Returns
    -------
    pd.DataFrame
        df with columns 'date' and a new name based on info_data for cleaned serie
    """
    if info_data.get("series_id") is None:
        raise KeyError("No 'series_id' in info_data: this key is mandatory.")

    data = pd.json_normalize(obs_data["observations"])
    data.drop(columns=["realtime_start", "realtime_end"], inplace=True)
    name_components = []
    for field in [
        "series_id",
        "frequency",
        "units",
        "aggregation_method",
        "seasonal_adjustment",
    ]:
        if info_data.get(field):
            name_components.append(info_data[field])
    series_name = "_".join(name_components)
    data.rename(columns={"value": series_name}, inplace=True)
    return data


def get_usbls_data(api_key: str, series_ids: list, start_date: str = None, end_date: str = None):
    """Get multiple series data and info from fred api
    and group in two lists (one info, one data)

    Parameters
    ----------
    api_key: str
        Your api key to access fred api
        see
    series_ids: list
        List of ids US BLS series
    start_date: str
        The start of the observation period. YYYY-MM-DD or YYYY
        USBLS only sends data for the whole year
    end_date: str
        The start of the observation period. YYYY-MM-DD or YYYY
        USBLS only sends data for the whole year

    Returns
    -------
    info_list: list
        List of dictionaries with metadata for every series retrieved
    data_list: list
        List of dictionaries with data reponse from fred api in json format
    """
    headers = {'Content-type': 'application/json'}
    payload = {
        "registrationkey": api_key,
        "catalog": True,
        "seriesid": series_ids
    }

    # years limited at 20 per query on usbls API v2
    # build 20 yeard periods, query for each one
    start_year, end_year = int(start_date[:4]), int(end_date[:4])
    periods = []
    while end_year - start_year >= 20:
        periods.append((str(start_year), str(start_year + 19)))
        start_year += 20
    periods.append((str(start_year), str(end_year)))

    obs_dict = defaultdict(list)
    for start_year, end_year in periods:
        payload.update({"startyear": start_year, "endyear": end_year})
        response = usbls_api_query(
            url=API_ENDPOINTS["USBLS_API_URL"],
            payload=payload,
            headers=headers
        )
        data = response["Results"]["series"]
        for i in range(len(series_ids)):
            obs_dict[data[i]["seriesID"]] += data[i]["data"]
    obs_list = [obs_dict[k] for k in series_ids]
    
    info_list = []
    for i, series_id in enumerate(series_ids):
        info_dict = {
            "name": data[i]["catalog"]["survey_name"],
            "series_id": series_id,
            "frequency": None,
            "units": None,
            "aggregation_method": None,
            "seasonal_adjustment": re.sub(
                r'[^NSA]', r'', data[i]["catalog"]["seasonality"]
            ).lower(),
        }
        info_list.append(info_dict)
        
    return obs_list, info_list
        

def usbls_api_query(url: str, payload: dict, headers: dict =None):
    """Send a post request to US BLS API

    Raises
    ------
    HTPPError

    Returns
    -------
    requests.Response.json()
    """
    payload = json.dumps(payload)
    response = requests.post(url=url, data=payload, headers=headers)
    response.raise_for_status()
    response = response.json()
    if response["status"] == "REQUEST_FAILED":
        raise requests.HTTPError("Message from US BLS API:\n" + "\n".join(response["message"]))
    if len(response["message"]) > 0:
        logger.warning("\n" + "\n".join(response["message"]))
    return response
