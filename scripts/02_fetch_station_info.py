import logging
from pathlib import Path

import pandas as pd

from utils.logging_config import setup_logging
from utils.tfl_api import api


URL = "https://api.tfl.gov.uk/StopPoint/Mode/tube"

TUBE_LINES = {
    "bakerloo",
    "central",
    "circle",
    "district",
    "hammersmith-city",
    "jubilee",
    "metropolitan",
    "northern",
    "piccadilly",
    "victoria",
    "waterloo-city",
}

# Fetches info about each station, we save id, as well as lines the station serves
def fetch_stations() -> pd.DataFrame:
    try:
        res = api(URL)

        rows = []

        # format response (station id, line id)
        for sp in res["stopPoints"]:
            if sp["stopType"] != "NaptanMetroStation":
                continue

            station = sp["stationNaptan"]

            for line in sp.get("lines", []):
                line_id = line["id"]

                if line_id not in TUBE_LINES:
                    continue

                rows.append({
                    "station_naptan": station,
                    "line_id": line_id,
                })

        return pd.DataFrame(rows).drop_duplicates()

    except Exception as e:
        logging.error(f"Failed to fetch stations: {e}")
        raise


if __name__ == "__main__":

    setup_logging()

    df = fetch_stations()
    # save to parquet
    OUT_DIR = Path(f"data/raw/stations/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_FILE = OUT_DIR / "line_station.parquet"

    df.to_parquet(OUT_FILE, index=False)

    logging.info(f"Successfully saved {len(df)} rows to {OUT_FILE}")
