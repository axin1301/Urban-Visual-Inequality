#!/usr/bin/env python3
"""Download ACS 5-year block-group estimates with diagnostic API failures."""
import argparse
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

# ACS detailed table: household median income (dollars) and MOE.
VARIABLES = ["NAME", "B19013_001E", "B19013_001M"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--states", nargs="+", required=True, help="Two-digit state FIPS codes, e.g. 36 17")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise SystemExit("CENSUS_API_KEY is required. Request a free key at https://api.census.gov/data/key_signup.html and export CENSUS_API_KEY=...")
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    records = []
    for state in args.states:
        state = str(state).zfill(2)
        url = f"https://api.census.gov/data/{args.year}/acs/acs5"
        params = {"get": ",".join(VARIABLES), "for": "block group:*",
                  "in": f"state:{state} county:* tract:*"}
        params["key"] = api_key
        try:
            resp = session.get(url, params=params, timeout=(15, 180))
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Census API connection failed for state {state}: {exc}. "
                "If running on LUMI, check outbound HTTPS connectivity/proxy settings."
            ) from exc
        # Never print the request URL: it includes the private API key.
        content_type = resp.headers.get("Content-Type", "unknown")
        if not resp.ok:
            raise RuntimeError(
                f"Census API HTTP {resp.status_code} for state {state}; "
                f"content-type={content_type}; response={resp.text[:500]!r}"
            )
        try:
            rows = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Census API response is not JSON for state {state}: "
                f"HTTP {resp.status_code}, content-type={content_type}, "
                f"body_prefix={resp.text[:500]!r}. "
                "Possible HTML proxy/login page, transient upstream error, "
                "or an invalid API response."
            ) from exc
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], list):
            raise RuntimeError(
                f"Unexpected Census API JSON structure for state {state}: "
                f"{str(rows)[:500]}"
            )
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
