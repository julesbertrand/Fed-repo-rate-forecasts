import datetime as dt
import pytest
import requests
from config.config import API_ENDPOINTS


@pytest.fixture(autouse=True)
def mock_response_get(monkeypatch):
    """requests.get mocked to return something based on MockAPIResponse class"""

    def mock_get(*args, **kwargs):
        return MockAPIResponse(*args, **kwargs)

    monkeypatch.setattr(requests, "get", mock_get)


class MockAPIResponse:
    """Produces a api mock response object for get requests"""

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.today_date = dt.date.today().strftime("%Y-%m-%d")

    @staticmethod
    def raise_for_status():
        pass

    def json(self):
        """Return json formated response"""
        if self.url == API_ENDPOINTS["FRED_API_URL_SER"]:
            return self.fred_api_response_ser(self.kwargs["params"])
        if self.url == API_ENDPOINTS["FRED_API_URL_OBS"]:
            return self.fred_api_response_obs(self.kwargs["params"])
        if self.url == API_ENDPOINTS["USBLS_API_URL"]:
            return self.usbls_api_response(self.kwargs["data"])
        raise RuntimeError(
            "Network access not allowed during testing! "
            f"No mock response for this url: {self.url}"
        )

    def fred_api_response_obs(self, params):
        """ fred api response format for obesrvations"""
        obs_val_dict = {
            "realtime_start": self.today_date,
            "realtime_end": self.today_date,
            "date": "1980-01-01",
            "value": "13.82",
        }
        resp = {
            "realtime_start": self.today_date,
            "realtime_end": self.today_date,
            "observation_start": params.get("observation_start"),
            "observation_end": params.get("observation_end"),
            "units": params.get("units", "lin"),
            "output_type": 1,
            "file_type": params.get("file_type"),
            "order_by": "observation_date",
            "sort_order": "asc",
            "count": 3,
            "offset": 0,
            "limit": 100000,
            "observations": [obs_val_dict],
        }
        return resp

    def fred_api_response_ser(self, params):
        """fred api response format for obesrvations"""
        resp = {
            "realtime_start": self.today_date,
            "realtime_end": self.today_date,
            "seriess": [
                {
                    "id": "FEDFUNDS",
                    "realtime_start": self.today_date,
                    "realtime_end": self.today_date,
                    "title": "Mock series name",
                    "observation_start": params.get("observation_start"),
                    "observation_end": params.get("observation_end"),
                    "notes": "mock notes",
                }
            ],
        }
        return resp

    def usbls_api_response(self, seriesids):
        raise NotImplementedError


@pytest.fixture
def expected_result_get_fred_data():
    result = (
        [
            {
                "realtime_start": dt.date.today().strftime("%Y-%m-%d"),
                "realtime_end": dt.date.today().strftime("%Y-%m-%d"),
                "observation_start": "1980-01-08",
                "observation_end": None,
                "units": "lin",
                "output_type": 1,
                "file_type": "json",
                "order_by": "observation_date",
                "sort_order": "asc",
                "count": 3,
                "offset": 0,
                "limit": 100000,
                "observations": [
                    {
                        "realtime_start": dt.date.today().strftime("%Y-%m-%d"),
                        "realtime_end": dt.date.today().strftime("%Y-%m-%d"),
                        "date": "1980-01-01",
                        "value": "13.82",
                    },
                ],
            },
            {
                "realtime_start": dt.date.today().strftime("%Y-%m-%d"),
                "realtime_end": dt.date.today().strftime("%Y-%m-%d"),
                "observation_start": "1980-01-08",
                "observation_end": None,
                "units": "lin",
                "output_type": 1,
                "file_type": "json",
                "order_by": "observation_date",
                "sort_order": "asc",
                "count": 3,
                "offset": 0,
                "limit": 100000,
                "observations": [
                    {
                        "realtime_start": dt.date.today().strftime("%Y-%m-%d"),
                        "realtime_end": dt.date.today().strftime("%Y-%m-%d"),
                        "date": "1980-01-01",
                        "value": "13.82",
                    },
                ],
            },
        ],
        [
            {
                "name": "Mock series name",
                "series_id": "FEDFUNDS",
                "frequency": "m",
                "units": "lin",
                "aggregation_method": None,
                "seasonal_adjustment": None,
            },
            {
                "name": "Mock series name",
                "series_id": "DFF",
                "frequency": "m",
                "units": "lin",
                "aggregation_method": "eop",
                "seasonal_adjustment": None,
            },
        ],
    )
    return result
