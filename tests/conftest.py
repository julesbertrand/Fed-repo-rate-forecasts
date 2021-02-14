import pytest
import requests
import datetime as dt
from config.config import API_URLS


@pytest.fixture(autouse=True)
def mock_response_get(monkeypatch):
    """requests.get mocked to return something based on MockAPIResponse class"""

    def mock_get(*args, **kwargs):
        return MockAPIResponse(*args, **kwargs)

    monkeypatch.setattr(requests, "get", mock_get)


class MockAPIResponse:
    def __init__(self, url, params, **kwargs):
        self.url = url
        self.params = params
        self.kwargs = kwargs
        self.today_date = dt.date.today().strftime("%Y-%m-%d")

    @staticmethod
    def raise_for_status():
        pass

    def json(self):
        if self.url == API_URLS["FRED_API_URL_SER"]:
            return self.fred_api_response_ser(self.params)
        elif self.url == API_URLS["FRED_API_URL_OBS"]:
            return self.fred_api_response_obs(self.params)
        elif self.url == API_URLS["USBLS_API_URL"]:
            return {"mock_key": "mock response from usbls"}
        else:
            raise RuntimeError(
                f"Network access not allowed during testing! No mock response for this url: {self.url}"
            )

    def fred_api_response_obs(self, params):
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