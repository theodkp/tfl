import requests
import os
TFL_APP_KEY = os.environ["TFL_APP_KEY"]


def api(url: str, params=None):
    if params is None:
        params = {}

    params["app_key"] = TFL_APP_KEY

    resp = requests.get(
        url,
        params=params,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
