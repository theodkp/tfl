import json
import requests
import pandas as pd

# todo
def api(URL,KEY):
    resp = requests.get(URL, headers={"Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    return data