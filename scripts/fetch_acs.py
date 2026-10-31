#!/usr/bin/env python3
"""Download ACS 5-year block-group estimates. Uses optional CENSUS_API_KEY."""
import argparse
import os
from pathlib import Path
import requests
import pandas as pd

# ACS detailed table: household median income (dollars) and MOE.
VARIABLES = ["NAME", "B19013_001E", "B19013_001M"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--states", nargs="+", required=True, help="Two-digit state FIPS codes, e.g. 36 17")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    records = []
    for state in args.states:
        state = str(state).zfill(2)
        url = f"https://api.census.gov/data/{args.year}/acs/acs5"
        params = {"get": ",".join(VARIABLES), "for": "block group:*",
                  "in": f"state:{state} county:* tract:*"}
        if os.getenv("CENSUS_API_KEY"):
            params["key"] = os.environ["CENSUS_API_KEY"]
        resp = requests.get(url, params=params, timeout=180)
        resp.raise_for_status()
        rows = resp.json()
        if len(rows) < 2:
            raise RuntimeError(f"No block groups returned for state {state}")
        records += [dict(zip(rows[0], row)) for row in rows[1:]]
        print(f"{state}: {len(rows)-1} block groups")
    df = pd.DataFrame(records)
    df["geoid"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    if not df["geoid"].str.fullmatch(r"\d{12}").all():
        raise ValueError("Invalid block group GEOID")
    for old, new in [("B19013_001E", "median_household_income"),
                     ("B19013_001M", "income_moe90")]:
        df[new] = pd.to_numeric(df[old], errors="coerce")
        df.loc[df[new] < 0, new] = float("nan")  # Census missing/suppressed codes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["geoid","NAME","state","county","tract","block group",
        "median_household_income","income_moe90"]].to_csv(args.out,index=False)
    print(f"Saved {len(df)} block groups to {args.out}")

if __name__ == "__main__":
    main()
