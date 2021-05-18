from frp.utils.path import get_projet_root

### Path ###
ROOT_PATH = get_projet_root()


### Data Getters ###
API_KEYS_FILEPATH = "./secrets/api_keys.yaml"
API_ENDPOINTS = {
    "FRED_OBS": "https://api.stlouisfed.org/fred/series/observations?",
    "FRED_SER": "https://api.stlouisfed.org/fred/series?",
    "FRED": "https://api.stlouisfed.org/fred/",
    "USBLS": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
    "OECD": "http://stats.oecd.org/SDMX-JSON/data/",
}
API_REQUESTS_PARAMS_FILEPATH = "./config/config_api.yaml"


### Get data ###
GETTERS = ["FRED", "USBLS", "OECD"]
