#!/usr/bin/env python3
"""One-city, SQLite-backed SAT/STV/FUSION pilot. BLOBs fetched only when needed."""
import argparse
import csv
import hashlib
import json
import random
import sqlite3
from contextlib import closing
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"


def connect_ro(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def sample_ids(ids, maximum, seed, key):
    ids = sorted(set(ids), key=str)
    rng = random.Random(int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:16], 16))
    return sorted(rng.sample(ids, min(len(ids), maximum)), key=str)


def all_cbgs(sat_db, stv_db, city, lookup=None):
    """City geographic lookup defines full universe when provided; otherwise union of DB CBGs."""
    pool = set()
    if lookup:
        import pandas as pd
        df = pd.read_csv(lookup, dtype=str, encoding="utf-8-sig")
        hit = df.loc[df["City Name"].str.strip().eq(city), "CBG Code"].dropna()
        pool.update(hit.str.strip().tolist())
    else:
        for db in (sat_db, stv_db):
            pool.update(str(r[0]) for r in db.execute(
                "SELECT DISTINCT cbg_code FROM cbg WHERE city_name=?", (city,)
            ))
    for geoid in sorted(pool):
        if len(geoid) != 12 or not geoid.isdigit():
            raise ValueError(f"Expected 12-digit block-group CBG Code; got {geoid!r}")
        yield geoid


def sat_candidates(db, city, geoid, year=None):
    q = """SELECT c.image_id, c.year, c.source_year
           FROM cbg c WHERE c.city_name=? AND c.cbg_code=?
             AND c.status='ready'
             AND EXISTS(SELECT 1 FROM images i WHERE i.image_id=c.image_id)"""
    params = [city, geoid]
    if year is not None:
        q += " AND c.year=?"
        params.append(year)
    return [dict(image_id=x, year=y, source_year=z) for x,y,z in db.execute(q,params)]


def stv_candidates(db, city, geoid, year=None):
    q = """SELECT DISTINCT v.view_id, v.image_id, v.latitude, v.longitude,
                  c.year, c.source_year
           FROM cbg c
           JOIN view_cbg m ON m.cbg_id=c.cbg_id
           JOIN views v ON v.view_id=m.view_id
           WHERE c.city_name=? AND c.cbg_code=?
             AND EXISTS(SELECT 1 FROM images i WHERE i.view_id=v.view_id)"""
    params = [city, geoid]
    if year is not None:
        q += " AND c.year=?"
        params.append(year)
    q += " ORDER BY v.view_id"
    return [dict(view_id=x, panorama_id=y, latitude=lat, longitude=lon,
                 year=yr, source_year=src) for x,y,lat,lon,yr,src in db.execute(q,params)]


def load_image(db, modality, record, edge):
    from PIL import Image, ImageOps
    if modality == "SAT":
        row=db.execute("SELECT png FROM images WHERE image_id=?", (record["image_id"],)).fetchone()
    else:
        row=db.execute("SELECT image FROM images WHERE view_id=?", (record["view_id"],)).fetchone()
    if row is None:
        raise FileNotFoundError(f"{modality} record missing: {record}")
    with Image.open(BytesIO(row[0])) as opened:
        image=ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((edge,edge))
        return image.copy()


def model_setup(config):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. On ROCm, PyTorch still uses torch.cuda APIs.")
    processor=AutoProcessor.from_pretrained(config["model_id"],trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        config["model_id"],torch_dtype=getattr(torch,config["dtype"]),
        device_map=config["device_map"],trust_remote_code=False)
    return model.eval(),processor


def infer(model, processor, condition, style, sats, stvs, config):
    import torch
    # Keep each image physically present in the same batch and mark its modality.
    content=[]
    if sats:
        if condition=="FUSION": content.append({"type":"text","text":"SET A: SATELLITE"})
        for im in sats: content.append({"type":"image","image":im})
    if stvs:
        if condition=="FUSION": content.append({"type":"text","text":"SET B: STREET VIEW"})
        for im in stvs: content.append({"type":"image","image":im})
    content.append({"type":"text","text":(PROMPTS/f"{style}_{condition.lower()}.txt").read_text()})
    messages=[{"role":"system","content":(PROMPTS/"system.txt").read_text()},
              {"role":"user","content":content}]
    inputs=processor.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    # In auto device-map mode, input tensors start on the first parameter's device.
    inputs=inputs.to(next(model.parameters()).device)
    with torch.inference_mode():
        outputs=model.generate(**inputs, do_sample=False,
                               max_new_tokens=config["max_new_tokens"])
    return processor.decode(outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--city",required=True,help="Exact city_name, e.g. 'Boston city'")
    p.add_argument("--sat-db",type=Path,required=True)
    p.add_argument("--stv-db",type=Path,required=True)
    p.add_argument("--lookup",type=Path,help="Full Geographic_Lookup_Table_Between_City_CBG.csv")
    p.add_argument("--config",type=Path,default=ROOT/"configs/pilot.json")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--year",type=int,default=None,help="Optional filter on cbg.year in BOTH DBs")
    p.add_argument("--max-cbgs",type=int,default=0,help="0 = all eligible BGs; first N eligible for smoke test")
    p.add_argument("--paired-only",action="store_true",help="Only CBGs with both SAT and STV")
    p.add_argument("--dry-run",action="store_true",help="Do not fetch BLOBs or load a model")
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    # Resume ONLY successful tasks. Different config uses separate output paths.
    completed=set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    obj=json.loads(line)
                    if obj.get("status")=="ok" or (args.dry_run and obj.get("status")=="planned"):
                        completed.add(obj["key"])
    with closing(connect_ro(args.sat_db)) as sat_db, closing(connect_ro(args.stv_db)) as stv_db:
        cbgs=list(all_cbgs(sat_db,stv_db,args.city,args.lookup))
        # Metadata only: candidate IDs are cheap, and images remain in SQLite.
        tasks=[]
        eligible=0
        for geoid in cbgs:
            sat=sat_candidates(sat_db,args.city,geoid,args.year)
            stv=stv_candidates(stv_db,args.city,geoid,args.year)
            if args.paired_only and (not sat or not stv):
                continue
            eligible+=1
            if args.max_cbgs and eligible>args.max_cbgs:
                break
            sat_map={str(item["image_id"]):item for item in sat}
            stv_map={str(item["view_id"]):item for item in stv}
            sat_selected=[sat_map[v] for v in sample_ids(sat_map,cfg["max_sat_images"],cfg["seed"],geoid+":SAT")]
            stv_selected=[stv_map[v] for v in sample_ids(stv_map,cfg["max_stv_images"],cfg["seed"],geoid+":STV")]
            for condition in cfg["conditions"]:
                a=sat_selected if condition in ("SAT","FUSION") else []
                b=stv_selected if condition in ("STV","FUSION") else []
                if (condition=="SAT" and not a) or (condition=="STV" and not b) or (condition=="FUSION" and (not a or not b)):
                    continue
                for style in cfg["prompts"]:
                    key=f"{args.city}|{geoid}|{args.year}|{condition}|{style}"
                    if key not in completed:
                        tasks.append(dict(key=key,city=args.city,geoid=geoid,
                                          year_filter=args.year,condition=condition,
                                          prompt_type=style,sat_records=a,stv_records=b))
        print(f"City={args.city!r}; full_pool={len(cbgs)} CBGs; eligible_scanned={min(eligible,args.max_cbgs) if args.max_cbgs else eligible}; pending={len(tasks)} tasks",flush=True)
        if not tasks:return
        model=processor=None
        if not args.dry_run: model,processor=model_setup(cfg)
        # Cache images only within a single CBG; never materialize the full city.
        previous=None
        cached_sat=cached_stv=None
        with args.out.open("a",encoding="utf-8") as f:
            for i,t in enumerate(tasks,1):
                if t["geoid"]!=previous:
                    previous=t["geoid"]
                    cached_sat=cached_stv=None
                result={**t,"model_id":cfg["model_id"],"status":"planned" if args.dry_run else "error"}
                if not args.dry_run:
                    try:
                        if t["sat_records"] and cached_sat is None:
                            cached_sat=[load_image(sat_db,"SAT",x,cfg["max_image_edge"]) for x in t["sat_records"]]
                        if t["stv_records"] and cached_stv is None:
                            cached_stv=[load_image(stv_db,"STV",x,cfg["max_image_edge"]) for x in t["stv_records"]]
                        result["text"]=infer(model,processor,t["condition"],t["prompt_type"],
                            cached_sat if t["sat_records"] else [],
                            cached_stv if t["stv_records"] else [],cfg)
                        result["status"]="ok"
                    except Exception as exc:
                        result["error"]=f"{type(exc).__name__}: {exc}"
                f.write(json.dumps(result,ensure_ascii=False,default=str)+"\n")
                f.flush()
                if i%20==0 or i==len(tasks):
                    print(f"{i}/{len(tasks)} tasks; last={result['status']}",flush=True)


if __name__=="__main__":
    main()
