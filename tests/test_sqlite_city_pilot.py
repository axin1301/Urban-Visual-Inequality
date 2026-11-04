import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_sqlite_dry_run(tmp_path):
    sat=tmp_path/"sat.sqlite"
    stv=tmp_path/"stv.sqlite"
    with sqlite3.connect(sat) as c:
        c.executescript("""
        CREATE TABLE cbg(city_name TEXT,city_geoid TEXT,cbg_code TEXT,year INTEGER,
            source_year INTEGER,status TEXT,image_id TEXT);
        CREATE TABLE images(image_id TEXT,png BLOB);
        INSERT INTO cbg VALUES ('Boston city','25070','250250001001',2020,2020,'ready','sat1');
        INSERT INTO images VALUES ('sat1',x'89504e47');
        """)
    with sqlite3.connect(stv) as c:
        c.executescript("""
        CREATE TABLE cbg(cbg_id INTEGER,city_name TEXT,city_geoid TEXT,cbg_code TEXT,
            year INTEGER,source_year INTEGER);
        CREATE TABLE view_cbg(cbg_id INTEGER,view_id INTEGER);
        CREATE TABLE views(view_id INTEGER,image_id TEXT,latitude REAL,longitude REAL);
        CREATE TABLE images(view_id INTEGER,image BLOB);
        INSERT INTO cbg VALUES (1,'Boston city','25070','250250001001',2020,2020);
        INSERT INTO view_cbg VALUES (1,100);
        INSERT INTO views VALUES (100,'pano1',42.3,-71.0);
        INSERT INTO images VALUES (100,x'ffd8ff');
        """)
    out=tmp_path/"plan.jsonl"
    cmd=[sys.executable,str(ROOT/"scripts/pilot_sqlite.py"),"--city","Boston city",
         "--sat-db",str(sat),"--stv-db",str(stv),"--paired-only","--dry-run",
         "--out",str(out)]
    subprocess.run(cmd,check=True)
    subprocess.run(cmd,check=True)
    rows=[json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows)==6
    assert {r["condition"] for r in rows}=={"SAT","STV","FUSION"}
    assert {r["prompt_type"] for r in rows}=={"open","standardized"}
    assert all(r["geoid"]=="250250001001" for r in rows)
