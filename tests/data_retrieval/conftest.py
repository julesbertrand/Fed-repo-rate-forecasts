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
        if self.url == "test_url":
            raise requests.HTTPError("Test Error")


def load_test_data(filepath):
    """Load test data from json file"""
    with open(filepath, "r") as data:
        test_data = yaml.safe_load(data)
    return test_data


def pytest_generate_tests(metafunc):
    if "test_data_give_name_to_series" in metafunc.fixturenames:
        test_data = load_test_data("tests/data_retrieval/test_data_give_name_to_series.yaml")
        test_data_list = zip(*test_data)
        metafunc.parametrize("test_data_give_name_to_series", test_data_list)
