# Qwen3-VL pilot (100-city SAT / STV)

This is a **pilot scaffold**, not a completed or validated model run. No vLLM, CUDA-only kernels, or external inference API is assumed.

## Input format

One CSV per city in `metadata/`. CSV filename must match the city folder name in `images/`. Each row represents one 12-digit **2020 block-group GEOID** (as a string).

```csv
geoid,sat_images,stv_images
360610001001,"[""a.png"",""b.png""]","[""a.jpg"",""b.jpg""]"
```

Folders: `images/CityName/sat/a.png` and `images/CityName/stv/a.jpg`. Any boundary columns are ignored by the pilot. Array cells may contain JSON lists or Python literal lists. All filenames must be relative paths within the appropriate modality directory; no image files are committed to Git.

If your existing columns differ, pass `--geoid-col`, `--sat-col`, and `--stv-col`. If your city files use different filenames from directory names, make compatible symlinks or extend the loader. **Do not send ACS labels in model inputs.**

## Dry run without GPU

```bash
python scripts/pilot.py --metadata /path/to/city_csvs \
  --image-root /path/to/images --out results/plan.jsonl --dry-run --limit 12
python -m pytest -q tests/test_pilot.py
```

The dry run writes planned JSONL records only; it does not load images or check that actual image files exist. Run once with a separate output filename for inference (e.g., `results/pilot.jsonl`) so planned and real records do not mingle.

## LUMI ROCm preflight

LUMI GPUs are AMD and require a compatible ROCm PyTorch/Transformers environment. Install `requirements-pilot.txt` **inside** a validated LUMI-supported PyTorch environment; do not replace ROCm PyTorch with a generic pip CUDA build.

```bash
python scripts/check_rocm.py
python scripts/pilot.py --metadata /path/to/city_csvs \
  --image-root /path/to/images --out results/pilot.jsonl --limit 1
```

First check your allocated GPU architecture, memory, dtype availability, model implementation, attention kernels, automatic device placement and multi-image memory. 32B parameters at BF16 alone occupy approximately 64 GB before activations and KV cache; long multi-image prompts can substantially increase usage. A passing preflight does **not** mean Qwen3-VL-32B runs on LUMI. The default Slurm script is a **template**: check allocation, partition, GPU count and ROCm environment before use.

After verifying one task, increase `--limit` and use per-job JSONL output paths to prevent simultaneous writes. The script resumes completed `status=ok` records when reusing an output file, and permits retry of failed records. Records are append-only. `--limit` limits **tasks** (CBG × condition × prompt), not block groups.

## Experimental design

- Frozen `Qwen/Qwen3-VL-32B-Instruct`, greedy decoding, three modalities SAT/STV/FUSION and two prompts Open-ended/Standardized.
- Fixed per-CBG sample of up to four SAT images and eight STV images. SHA256-based selection is deterministic across runs.
- Keep raw responses in JSONL along with GEOID, prompt type and selected image paths.
- SAT/STV use potentially different image counts: do image-count and token-budget sensitivity experiments before comparing inequality magnitudes.
- Review original image licensing; image timestamps, clouds, color processing and sample coverage require auditing.
- ACS socioeconomic labels remain a separate table, joined **only after** generation.
- Cross-model robustness, human calibration, text embedding, ACS join and inequality analyses are not implemented in this pilot.

## Important limitations

The pilot loads images using PIL, converting to RGB and resizing to a 1024-pixel maximum edge. This works with typical PNG/JPEG and some TIFF exports, but **multiband Sentinel TIFFs must be rendered to RGB first**. Many views plus a 32B model may exceed available GPU memory. The inference backend uses Transformers `AutoModelForImageTextToText` rather than assuming vLLM; exact installed Transformers/Qwen version must be validated on your LUMI node.
