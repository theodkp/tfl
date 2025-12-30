import pandas as pd
from src.tfl_api import api
from pathlib import Path
import logging
import os


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


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def fetch_stations() -> pd.DataFrame:
    ''' Fetches info about each station, we save id as well as lines the station serves '''


    try:
        res = api(URL)

        rows = []

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



    df = fetch_stations()

    OUT_DIR = Path(f"data/raw/stations/")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_FILE = OUT_DIR / "line_station.parquet"

    df.to_parquet(OUT_FILE, index=False)

    logging.info(f"Successfully saved {len(df)} rows to {OUT_FILE}")


