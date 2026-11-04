# Single-city SQLite-backed pilot

## Why this replaces the CSV image paths

The research imagery is stored as **PNG/JPEG BLOBs** in two existing SQLite databases, not as standalone files. This pipeline queries the CBG-to-image relations, selects image IDs, and decodes only selected BLOBs at inference time. It does not export images or require road segmentation.

Existing schema assumptions based on the provided extract code:
- SAT: `cbg(city_name,city_geoid,cbg_code,year,source_year,status,image_id)`, `images(image_id,png)`.
- STV: `cbg(cbg_id,city_name,city_geoid,cbg_code,year,source_year)`, `view_cbg(cbg_id,view_id)`, `views(view_id,image_id,latitude,longitude)`, `images(view_id,image)`.
- Optional lookup CSV: `City Name` and `CBG Code` columns. This preserves the *complete* city CBG pool, including CBGs with no imagery.

This has **not yet been verified against the real DB**. Query errors should be reported; do not modify original databases. SQLite connections use read-only URI mode.

## 1. Update and preflight

```bash
git pull origin main
python scripts/check_rocm.py  # GPU allocation, not login node, for GPU availability
```

## 2. Dry run: Boston city

```bash
export UVI_ROOT=/scratch/project_462001391/xiyanxin/Urban_Visual_Inequality
python scripts/pilot_sqlite.py \
  --city "Boston city" \
  --sat-db "$UVI_ROOT/CBG_satellite_z15_black.sqlite" \
  --stv-db "$UVI_ROOT/stv_images.sqlite" \
  --lookup "$UVI_ROOT/STV-image-statistics/CBD-city-data/Basic_Geographic_Statistics/Geographic_Lookup_Table_Between_City_CBG.csv" \
  --paired-only --max-cbgs 5 --dry-run \
  --out results/boston_plan.jsonl
```

This does not load any image BLOBs, model, or GPU. It should generate **30 planned tasks** if 5 CBGs have both modalities: SAT, STV, FUSION × open, standardized. `--max-cbgs` counts eligible CBGs, not task rows. Output retains GEOIDs and IDs; no image bytes.

## 3. Tiny real inference, then all Boston CBGs

In a verified ROCm-compatible PyTorch/Transformers environment on a **GPU compute node**, test five eligible CBGs:

```bash
python scripts/pilot_sqlite.py \
  --city "Boston city" \
  --sat-db "$UVI_ROOT/CBG_satellite_z15_black.sqlite" \
  --stv-db "$UVI_ROOT/stv_images.sqlite" \
  --lookup "$UVI_ROOT/STV-image-statistics/CBD-city-data/Basic_Geographic_Statistics/Geographic_Lookup_Table_Between_City_CBG.csv" \
  --paired-only --max-cbgs 5 \
  --out results/boston_inference.jsonl
```

When this has been validated, **use a new output file** for the full-city run, removing `--max-cbgs`:

```bash
python scripts/pilot_sqlite.py \
  --city "Boston city" \
  --sat-db "$UVI_ROOT/CBG_satellite_z15_black.sqlite" \
  --stv-db "$UVI_ROOT/stv_images.sqlite" \
  --lookup "$UVI_ROOT/STV-image-statistics/CBD-city-data/Basic_Geographic_Statistics/Geographic_Lookup_Table_Between_City_CBG.csv" \
  --paired-only \
  --out results/boston_full.jsonl
```

By default the script does **not** restrict the image year because satellite/STV capture years can differ from the indicator year. Option `--year 2020` filters **CBG year in both databases**, which may exclude most images: use only after inspecting year distribution. The JSONL includes source year metadata for later mismatch checks.

The output supports resumability: a task is considered completed only with `status=ok`; failures can be retried. Avoid multiple processes appending to the same JSONL. Config changes, prompts or model changes require a new output file to avoid mixed experimental conditions.

## 4. ACS: Boston is Massachusetts, not New York/Illinois

The prior `data/acs_2024.csv` for states 36 and 17 does **not** cover Boston. Massachusetts state FIPS is **25**.

```bash
python scripts/fetch_acs.py --year 2024 --states 25 --out data/acs_ma_2024.csv
python scripts/analyze_city_pilot.py \
  --results results/boston_full.jsonl \
  --acs data/acs_ma_2024.csv \
  --out results/boston_analysis
```

Outputs: `completion_summary.csv`, `cbg_text_metrics.csv`, `income_group_summary.csv`. These are feasibility and SES-stratified text statistics, **not** ground-truth visual inequality measures.

Optional **exploratory** embedding+income prediction probe:

```bash
pip install sentence-transformers scikit-learn
python scripts/analyze_city_pilot.py \
  --results results/boston_full.jsonl --acs data/acs_ma_2024.csv \
  --out results/boston_analysis --embed --device cpu
```

Output: `exploratory_embedding_probe.csv` with 5-fold random-CV log-income R² and MAE. This is a smoke test, **not a defensible causal or spatially independent test**. Publishable analyses require geographic splits and evidence validation.

## 5. Additional checks before large batches

- Confirm `CBG Code` is a 12-digit Census block-group GEOID and that it matches the ACS geography vintage. 
- Inspect `year`/`source_year`, SAT resolution/color handling, cloud cover and STV duplicate panoramas.
- Inspect CBG with only SAT or only STV separately; `--paired-only` deliberately excludes them from the main balanced experiment.
- Larger 32B multi-image prompts may OOM. LUMI's ROCm/GPU configuration must be tested, do not assume CUDA/vLLM compatibility.
- The current pipeline uses Transformers with **one inference process**, and does not implement Slurm array sharding or tensor parallel serving. Those need validation before running a large-city job.
