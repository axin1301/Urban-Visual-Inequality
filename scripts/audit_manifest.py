#!/usr/bin/env python3
"""Audit satellite and street-view coverage against ACS, without road data."""
import argparse
from pathlib import Path
import pandas as pd

def read_csv(path):
    return pd.read_csv(path, dtype={"geoid": str}, keep_default_na=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--acs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--check-files", action="store_true")
    args = p.parse_args()
    m = read_csv(args.manifest)
    a = read_csv(args.acs)
    required = {"geoid", "city", "modality", "image_path"}
    if missing := required - set(m.columns):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if not {"geoid","median_household_income"}.issubset(a.columns):
        raise ValueError("ACS file must include geoid and median_household_income")
    m["geoid"] = m["geoid"].str.strip()
    if not m["geoid"].str.fullmatch(r"\d{12}").all():
        raise ValueError("GEOID must be a 12-digit block-group string; preserve leading zeros.")
    m["modality"] = m["modality"].str.upper().str.strip()
    if not m["modality"].isin(["SAT","STV"]).all():
        raise ValueError("modality must be SAT or STV")
    if "image_year" not in m:
        m["image_year"] = pd.NA
    m["image_year"] = pd.to_numeric(m["image_year"], errors="coerce")
    m["file_exists"] = m["image_path"].map(lambda x: Path(str(x)).is_file()) if args.check_files else pd.NA
    counts = m.groupby(["city","geoid","modality"]).size().unstack(fill_value=0).reset_index()
    for v in ("SAT","STV"):
        if v not in counts: counts[v] = 0
    # Each BG is counted once; repeated imagery cannot artificially increase matching rate.
    counts["both_modalities"] = (counts["SAT"] > 0) & (counts["STV"] > 0)
    acs = a[["geoid","median_household_income"]].drop_duplicates("geoid").copy()
    acs["median_household_income"] = pd.to_numeric(acs["median_household_income"],errors="coerce")
    counts = counts.merge(acs,on="geoid",how="left",validate="many_to_one")
    counts["acs_available"] = counts["median_household_income"].notna()
    counts["usable"] = counts["both_modalities"] & counts["acs_available"]
    # Within-city SES quintiles, on observed ACS labels only.
    def quantile(s):
        out = pd.Series(pd.NA, index=s.index, dtype="Int64")
        valid = s.dropna()
        if valid.nunique() >= 5:
            out.loc[valid.index] = pd.qcut(valid.rank(method="first"),5,labels=False).astype(int)+1
        return out
    counts["income_quintile"] = counts.groupby("city")["median_household_income"].transform(quantile)
    by_city = counts.groupby("city",dropna=False).agg(
        block_groups=("geoid","size"),
        sat_bg=("SAT",lambda s: int((s>0).sum())),
        stv_bg=("STV",lambda s: int((s>0).sum())),
        paired_bg=("both_modalities","sum"),
        paired_with_acs=("usable","sum"),
        income_available=("acs_available","sum")).reset_index()
    by_income = counts.groupby(["city","income_quintile"],dropna=False).agg(
        observed_bg=("geoid","size"), paired_bg=("both_modalities","sum"),
        paired_with_acs=("usable","sum"),mean_stv_images=("STV","mean")).reset_index()
    by_income["paired_rate"] = by_income["paired_bg"] / by_income["observed_bg"]
    args.out.mkdir(parents=True,exist_ok=True)
    counts.to_csv(args.out/"blockgroup_coverage.csv",index=False)
    by_city.to_csv(args.out/"city_summary.csv",index=False)
    by_income.to_csv(args.out/"income_coverage.csv",index=False)
    if args.check_files:
        m.loc[~m["file_exists"],["geoid","modality","image_path"]].to_csv(args.out/"missing_files.csv",index=False)
    print(by_city.to_string(index=False))
    print(f"Saved audit to {args.out}")
    print("NOTE: coverage is among block groups present in the manifest, not all ACS block groups.")
    print("For population-wide coverage, join all ACS BGs with the manifest (including zero-image groups).")

if __name__ == "__main__":
    main()
