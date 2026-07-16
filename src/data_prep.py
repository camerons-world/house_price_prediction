from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

LAND_REGISTRY_DIR = DATA_RAW / "land_registry"
ONSPD_PATH = DATA_RAW / "onspd" / "onspd_full.csv"

PPD_URL_TEMPLATE = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-{year}.csv"
PPD_COLUMNS = [
    "transaction_id", "price", "date_of_transfer", "postcode", "property_type",
    "old_new", "duration", "paon", "saon", "street", "locality", "town_city",
    "district", "county", "ppd_category_type", "record_status",
]

# "greater london" = ppd's own county field, not a postcode-prefix guess
REGION_COUNTY = "GREATER LONDON"
YEARS = [2023, 2024, 2025]
DATE_START = "2023-01-01"
DATE_END = "2025-12-31"

OUTPUT_PATH = DATA_PROCESSED / "london_house_prices.csv"


def download_price_paid_year(year: int, dest_dir: Path = LAND_REGISTRY_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"pp-{year}.csv"
    if dest_path.exists():
        return dest_path
    tmp_path = dest_path.with_suffix(".csv.part")
    urllib.request.urlretrieve(PPD_URL_TEMPLATE.format(year=year), tmp_path)
    tmp_path.rename(dest_path)
    return dest_path


def _normalize_postcode(series: pd.Series) -> pd.Series:
    return series.str.strip().str.upper().str.replace(re.compile(r"\s+"), " ", regex=True)


def load_price_paid(years: list[int] = YEARS) -> pd.DataFrame:
    frames = []
    for year in years:
        path = download_price_paid_year(year)
        df = pd.read_csv(
            path, header=None, names=PPD_COLUMNS, quotechar='"',
            dtype={"postcode": "string"}, parse_dates=["date_of_transfer"],
        )
        frames.append(df[df["county"] == REGION_COUNTY])
    ppd = pd.concat(frames, ignore_index=True)
    ppd = ppd[(ppd["date_of_transfer"] >= DATE_START) & (ppd["date_of_transfer"] <= DATE_END)]
    ppd["postcode"] = _normalize_postcode(ppd["postcode"])
    return ppd.reset_index(drop=True)


def load_onspd_lookup(postcodes: set[str], onspd_path: Path = ONSPD_PATH) -> pd.DataFrame:
    # no stable direct-download url for onspd (arcgis item id changes per release) --
    # download manually from geoportal.statistics.gov.uk and place at onspd_path
    if not onspd_path.exists():
        raise FileNotFoundError(f"{onspd_path} not found -- download onspd csv manually")
    matches = []
    for chunk in pd.read_csv(
        onspd_path, usecols=["pcds", "lat", "long"], dtype={"pcds": "string"}, chunksize=250_000
    ):
        chunk["pcds"] = _normalize_postcode(chunk["pcds"])
        matches.append(chunk[chunk["pcds"].isin(postcodes)])
    lookup = pd.concat(matches, ignore_index=True).rename(columns={"pcds": "postcode"})
    return lookup.drop_duplicates(subset="postcode")


def build_dataset() -> pd.DataFrame:
    ppd = load_price_paid()
    lookup = load_onspd_lookup(set(ppd["postcode"]))
    return ppd.merge(lookup, on="postcode", how="inner")


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
