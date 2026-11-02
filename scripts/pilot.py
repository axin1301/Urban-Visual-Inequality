#!/usr/bin/env python3
"""Deterministic CBG SAT/STV/FUSION Qwen3-VL pilot; append-only JSONL."""
import argparse, ast, csv, hashlib, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def parse_list(value):
    if not value or value.strip().lower() in ("nan","none"): return []
    try: x=json.loads(value)
    except ValueError: x=ast.literal_eval(value)
    if not isinstance(x,list) or not all(isinstance(v,str) for v in x):
        raise ValueError("Image-list columns must contain arrays of filenames")
    return x

def get_images(cell,base,limit,seed,salt):
    base=base.resolve()
    found=[]
    for item in parse_list(cell):
        rel=Path(item)
        p=(base/rel).resolve()
        if rel.is_absolute() or not p.is_relative_to(base):
            raise ValueError("Only relative filenames inside each city modality directory are allowed")
        found.append(str(p))
    found=sorted(set(found))
    rng=random.Random(int(hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()[:16],16))
    return sorted(rng.sample(found,min(limit,len(found))))

def records(args,cfg):
    for csvfile in sorted(args.metadata.glob("*.csv")):
        city=csvfile.stem
        with csvfile.open(encoding="utf-8-sig",newline="") as f:
            reader=csv.DictReader(f)
            required={args.geoid_col,args.sat_col,args.stv_col}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{csvfile}: missing {required-set(reader.fieldnames or [])}")
            for line,row in enumerate(reader,2):
                geoid=str(row[args.geoid_col]).strip()
                if len(geoid)!=12 or not geoid.isdigit():
                    raise ValueError(f"{csvfile}:{line} invalid 12-digit GEOID {geoid!r}")
                sat=get_images(row[args.sat_col],args.image_root/city/cfg["sat_dirname"],cfg["max_sat_images"],cfg["seed"],geoid+":sat")
                stv=get_images(row[args.stv_col],args.image_root/city/cfg["stv_dirname"],cfg["max_stv_images"],cfg["seed"],geoid+":stv")
                for condition in cfg["conditions"]:
                    a=sat if condition in ("SAT","FUSION") else []
                    b=stv if condition in ("STV","FUSION") else []
                    if not a and not b: continue
                    # For fusion require both views.
                    if condition=="FUSION" and (not a or not b): continue
                    for prompt_type in cfg["prompts"]:
                        yield dict(key=f"{city}|{geoid}|{condition}|{prompt_type}",city=city,geoid=geoid,
                                   condition=condition,prompt_type=prompt_type,sat_images=a,stv_images=b)

def generate(model,processor,job,cfg):
    import torch
    from PIL import Image,ImageOps
    parts=[]
    for tag,paths in (("SET A: satellite images",job["sat_images"]),("SET B: street-view images",job["stv_images"])):
        if paths and job["condition"]=="FUSION":parts.append({"type":"text","text":tag})
        for file in paths:
            with Image.open(file) as im:
                image=ImageOps.exif_transpose(im).convert("RGB")
                image.thumbnail((cfg["max_image_edge"],cfg["max_image_edge"]))
                parts.append({"type":"image","image":image.copy()})
    prompt=(ROOT/"prompts"/f'{job["prompt_type"]}_{job["condition"].lower()}.txt').read_text()
    parts.append({"type":"text","text":prompt})
    messages=[
        {"role":"system","content":(ROOT/"prompts/system.txt").read_text()},
        {"role":"user","content":parts}
    ]
    features=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
    device=next(model.parameters()).device
    features=features.to(device)
    with torch.inference_mode():
        result=model.generate(**features,do_sample=False,max_new_tokens=cfg["max_new_tokens"])
    ids=result[0][features["input_ids"].shape[1]:]
    return processor.decode(ids,skip_special_tokens=True).strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata",type=Path,required=True)
    p.add_argument("--image-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--config",type=Path,default=ROOT/"configs/pilot.json")
    p.add_argument("--geoid-col",default="geoid")
    p.add_argument("--sat-col",default="sat_images")
    p.add_argument("--stv-col",default="stv_images")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    if not set(cfg["conditions"])<={"SAT","STV","FUSION"} or not set(cfg["prompts"])<={"open","standardized"}:
        raise ValueError("Invalid condition or prompt in config")
    if not list(args.metadata.glob("*.csv")):raise ValueError("No city CSVs found")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                v=json.loads(line)
                if v.get("status")=="ok" or (args.dry_run and v.get("status")=="planned"):
                    done.add(v["key"])
    pending=[r for r in records(args,cfg) if r["key"] not in done]
    if args.limit>0:pending=pending[:args.limit]
    print(f"Pending: {len(pending)}; dry-run={args.dry_run}",flush=True)
    if args.dry_run:
        with args.out.open("a") as f:
            for job in pending:f.write(json.dumps({**job,"status":"planned"})+"\n")
        return
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    if not torch.cuda.is_available():
        raise RuntimeError("No accelerator visible (ROCm exposes torch.cuda). Run on GPU allocation.")
    processor=AutoProcessor.from_pretrained(cfg["model_id"],trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(cfg["model_id"],torch_dtype=getattr(torch,cfg["dtype"]),device_map=cfg["device_map"],trust_remote_code=False)
    model.eval()
    with args.out.open("a") as f:
        for i,job in enumerate(pending,1):
            try:
                missing=[s for s in job["sat_images"]+job["stv_images"] if not Path(s).is_file()]
                if missing: raise FileNotFoundError(f"Missing image: {missing[0]}")
                text=generate(model,processor,job,cfg)
                record={**job,"model_id":cfg["model_id"],"status":"ok","text":text}
            except Exception as e:
                record={**job,"status":"error","error":f"{type(e).__name__}: {e}"}
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
            f.flush()
            print(i,len(pending),job["key"],record["status"],flush=True)

if __name__=="__main__":main()
