#!/usr/bin/env python3
"""First-pass city feasibility summary. Optional text-embedding SES probe (exploratory)."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CONDITIONS=["SAT","STV","FUSION"]

def load_results(path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line)
                if r.get("status")=="ok":
                    rows.append({k:r.get(k) for k in ("key","city","geoid","condition","prompt_type","text")})
    if not rows:
        raise ValueError("No completed status=ok records. Run real inference before analysis.")
    df=pd.DataFrame(rows)
    df["geoid"]=df["geoid"].astype(str)
    return df.drop_duplicates("key",keep="last")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--acs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--embed",action="store_true",help="Run optional English text embeddings + Ridge CV.")
    p.add_argument("--encoder",default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    df=load_results(args.results)
    acs=pd.read_csv(args.acs,dtype={"geoid":str})
    if not {"geoid","median_household_income"}.issubset(acs.columns):
        raise ValueError("ACS requires geoid and median_household_income")
    acs=acs.drop_duplicates("geoid")
    acs["median_household_income"]=pd.to_numeric(acs["median_household_income"],errors="coerce")
    acs.loc[acs["median_household_income"]<=0,"median_household_income"]=np.nan
    d=df.merge(acs[["geoid","median_household_income"]],on="geoid",how="left",validate="many_to_one")
    d["word_count"]=d["text"].fillna("").str.split().map(len)
    d["not_assessable_count"]=d["text"].fillna("").str.count("Not assessable")
    d["has_acs"]=d["median_household_income"].notna()
    d["income_quintile"]=pd.Series(pd.NA,index=d.index,dtype="Int64")
    for city, idx in d.groupby("city").groups.items():
        sub=d.loc[list(idx),["geoid","median_household_income"]].drop_duplicates("geoid")
        valid=sub.dropna()
        if valid["median_household_income"].nunique()>=5:
            # City-internal quintiles, derived once per GEOID.
            q=pd.qcut(valid["median_household_income"].rank(method="first"),5,labels=False).astype(int)+1
            lookup=dict(zip(valid["geoid"],q))
            d.loc[list(idx),"income_quintile"]=d.loc[list(idx),"geoid"].map(lookup).astype("Int64")
    d.drop(columns=["text"]).to_csv(args.out/"cbg_text_metrics.csv",index=False)
    summary=d.groupby(["city","condition","prompt_type"],dropna=False).agg(
        completed_tasks=("geoid","size"),unique_cbgs=("geoid","nunique"),
        acs_matched=("has_acs","sum"),mean_words=("word_count","mean"),
        mean_not_assessable=("not_assessable_count","mean")).reset_index()
    summary.to_csv(args.out/"completion_summary.csv",index=False)
    grp=d.groupby(["city","condition","prompt_type","income_quintile"],dropna=False).agg(
        n=("geoid","size"),mean_words=("word_count","mean"),
        mean_not_assessable=("not_assessable_count","mean")).reset_index()
    grp.to_csv(args.out/"income_group_summary.csv",index=False)
    print(summary.to_string(index=False))
    # Only compare modality results using the SAME CBGs for fair analysis.
    for style in d["prompt_type"].unique():
        part=d[d["prompt_type"]==style]
        complete=part.pivot_table(index=["city","geoid"],columns="condition",values="word_count",aggfunc="first")
        paired=complete.dropna(subset=CONDITIONS) if set(CONDITIONS).issubset(complete.columns) else complete.iloc[0:0]
        print(f"{style}: {len(paired)} CBGs have all 3 modalities")
    if args.embed:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.metrics import r2_score, mean_absolute_error
        model=SentenceTransformer(args.encoder,device=args.device)
        metrics=[]
        for (city,condition,style), sub in d[d["has_acs"]].groupby(["city","condition","prompt_type"]):
            if len(sub)<25:
                print(f"Skipping {city}/{condition}/{style}: fewer than 25 matched CBGs")
                continue
            x=model.encode(sub["text"].tolist(),batch_size=32,normalize_embeddings=True,show_progress_bar=False)
            y=np.log(sub["median_household_income"].to_numpy(dtype=float))
            # Exploratory random CV only, not spatially independent and not suitable for final inference.
            pred=cross_val_predict(Ridge(alpha=10.0),x,y,cv=KFold(n_splits=5,shuffle=True,random_state=42))
            metrics.append(dict(city=city,condition=condition,prompt_type=style,n=len(sub),
                                log_income_r2=r2_score(y,pred),
                                log_income_mae=mean_absolute_error(y,pred)))
        pd.DataFrame(metrics).to_csv(args.out/"exploratory_embedding_probe.csv",index=False)
        print("Embedding probe is exploratory: use spatial or held-out-city splits for publication.")
    print("Caveat: text lengths and 'Not assessable' counts do not prove true environmental omission or bias.")

if __name__=="__main__": main()
