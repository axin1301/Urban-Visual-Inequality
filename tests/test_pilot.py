import csv, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_dry_run(tmp_path):
    m=tmp_path/"meta";m.mkdir()
    img=tmp_path/"images"/"TestCity"
    (img/"sat").mkdir(parents=True)
    (img/"stv").mkdir()
    for p in ("sat/a.png","stv/b.jpg"): (img/p).write_bytes(b"dry-run-only")
    with (m/"TestCity.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["geoid","sat_images","stv_images"])
        w.writeheader();w.writerow({"geoid":"360610001001","sat_images":'["a.png"]',"stv_images":'["b.jpg"]'})
    out=tmp_path/"out.jsonl"
    cmd=[sys.executable,str(ROOT/"scripts/pilot.py"),"--metadata",str(m),"--image-root",str(tmp_path/"images"),"--out",str(out),"--dry-run"]
    subprocess.run(cmd,check=True);subprocess.run(cmd,check=True)
    assert len(out.read_text().splitlines())==6
