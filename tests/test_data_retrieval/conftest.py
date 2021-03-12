import json
import yaml
import pytest
import requests

from config.config import API_ENDPOINTS


@pytest.fixture(autouse=True)
def mock_response_get(monkeypatch):
    """requests.get mocked to return something based on MockAPIResponse class"""

    def mock_get(*args, **kwargs):
        return MockAPIResponse(*args, **kwargs)

    monkeypatch.setattr(requests, "get", mock_get)


@pytest.fixture(autouse=True)
def mock_response_post(monkeypatch):
    """requests.get mocked to return something based on MockAPIResponse class"""

    def mock_post(*args, **kwargs):
        return MockAPIResponse(*args, **kwargs)

    monkeypatch.setattr(requests, "post", mock_post)


class MockAPIResponse:
    """Produces a api mock response object for get requests"""

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.status_code = 200

    def raise_for_status(self):
        """Mock raise for status, raises HTTP error for url='test_url'"""
        if self.url == "test_url":
            raise requests.HTTPError("Test Error")

    def json(self):
        """Return json formated response"""
        if self.url == API_ENDPOINTS["FRED"] + "/series?":
            return self.fred_api_response_ser()
        if self.url == API_ENDPOINTS["FRED"] + "/series/observations?":
            return self.fred_api_response_obs()
        if self.url == API_ENDPOINTS["USBLS"]:
            return self.usbls_api_response()
        if API_ENDPOINTS["OECD"] in self.url:
            return self.oecd_api_response()
        raise NotImplementedError("This Getter Mock API Response does not exist")

    def fred_api_response_obs(self):
        """ fred api response format for observations"""
        params = self.kwargs.get("params")
        obs_val_dict = {
            "realtime_start": "2021-02-15",
            "realtime_end": "2021-02-15",
            "date": "1980-01-01",
            "value": "13.82",
        }
        response = {
            "realtime_start": "2021-02-15",
            "realtime_end": "2021-02-15",
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
        return response

    def fred_api_response_ser(self):
        """fred api response format for series metadata"""
        params = self.kwargs.get("params")
        response = {
            "realtime_start": "2021-02-15",
            "realtime_end": "2021-02-15",
            "seriess": [
                {
                    "id": "FEDFUNDS",
                    "realtime_start": "2021-02-15",
                    "realtime_end": "2021-02-15",
                    "title": "Mock series name",
                    "observation_start": params.get("observation_start"),
                    "observation_end": params.get("observation_end"),
                    "notes": "mock notes",
                    "units_short": "%",
                    "seasonal_adjustment_short": "NSA",
                }
            ],
        }
        return response

    def usbls_api_response(self):
        """us bls api mock response for post request"""
        data = json.loads(self.kwargs.get("data"))
        if data.get("seriesid") is None:
            response = {
                "status": "REQUEST_FAILED",
                "responseTime": 211,
                "message": ["Mock Error maessage from usbls"],
                "Results": {},
            }
            return response
        response = load_test_data("tests/samples/expected_usbls_mock_api_response.yaml")
        return response

    @staticmethod
    def oecd_api_response():
        """oecd api mock response for post request"""
        response = load_test_data("tests/samples/expected_oecd_mock_api_response.yaml")
        return response


def load_test_data(filepath):
    """Load test data from json file"""
    with open(filepath, "r") as data:
        test_data = yaml.load(data, Loader=yaml.Loader)
    return test_data


def pytest_generate_tests(metafunc):
    """Generates test parametrization for files for some tests"""
    if "test_data_give_name_to_series" in metafunc.fixturenames:
        test_data = load_test_data("tests/samples/test_data_give_name_to_series.yaml")
        test_data_list = zip(*test_data)
        metafunc.parametrize("test_data_give_name_to_series", test_data_list)


@pytest.fixture
def expected_result_get_fred_data():
    data = tuple(load_test_data("tests/samples/expected_result_get_fred_data.yaml"))
    return data


@pytest.fixture
def test_data_clean_fred_series():
    data = tuple(load_test_data("tests/samples/test_data_clean_fred_series.yaml"))
    return data


@pytest.fixture
def expected_result_get_usbls_data():
    data = tuple(load_test_data("tests/samples/expected_result_get_usbls_data.yaml"))
    return data
