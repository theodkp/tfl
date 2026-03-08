# polled every 5 minutes from aws ec2 instance

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.logging_config import setup_logging
from utils.tfl_api import api


URL = "https://api.tfl.gov.uk/Line/Mode/tube/Status"

   

# Fetches a single snapshot of tube line statuses 
def fetch_line_status_snapshot(snapshot_time: datetime) -> pd.DataFrame:

    try:
        res = api(URL)
        rows = []

        # format response (line id, line name, status severity, status description)
        for line in res:
            line_id = line["id"]
            line_name = line["name"]

            statuses = line.get("lineStatuses", [])
            if not statuses:
                continue

            status = statuses[0]

            rows.append({
                "snapshot_time_utc": snapshot_time,
                "line_id": line_id,
                "line_name": line_name,
                "status_severity": status["statusSeverity"],
                "status_description": status["statusSeverityDescription"],
            })

        return pd.DataFrame(rows)

    except Exception as e:
        logging.error(f"Failed to fetch line status: {e}")
        raise


if __name__ == "__main__":

    setup_logging()

    # time for filename
    snapshot_time = datetime.now(timezone.utc)

    df = fetch_line_status_snapshot(snapshot_time)

    date_str = snapshot_time.strftime("%Y-%m-%d")

    OUT_DIR = Path(f"data/raw/line_status/date={date_str}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_FILE = OUT_DIR / "snapshots.parquet"

    # if file exists, append to it
    if OUT_FILE.exists():
        existing = pd.read_parquet(OUT_FILE)
        df_final = pd.concat([existing, df], ignore_index=True)
    else:
        df_final = df

    temp_file = OUT_FILE.with_suffix(".tmp")
    df_final.to_parquet(temp_file, index=False)
    temp_file.replace(OUT_FILE)
    
    logging.info(f"Successfully saved {len(df)} rows to {OUT_FILE}")

    logging.debug(df_final.head(5))
