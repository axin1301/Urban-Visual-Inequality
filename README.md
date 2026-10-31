# Urban Visual Inequality

Research code for studying **visual inequality across US Census block groups** using satellite (SAT), street-view (STV), and ACS socioeconomic data.

## Research question

Do multimodal language models construct systematically unequal representations of urban environments, beyond differences in independently verifiable visual evidence?

The project draws inspiration from *Large language models create an uneven informational layer over cities* and *Invisible Walls in Cities*, but **does not** study recommendations, mobility, or sentiment. No road segmentation is required.

## Planned workflow

1. Index existing SAT and STV images by **12-digit block-group GEOID** (preserve leading zeros).
2. Retrieve ACS 5-year socioeconomic labels with estimate uncertainty.
3. Audit imagery coverage and temporal alignment by city and SES stratum.
4. Run frozen MLLM visual evidence tasks with SAT-only, STV-only and SAT+STV conditions, blinded to SES.
5. Validate with human annotation and compare conditional omissions, inconsistencies, and errors across SES groups.
6. Test robustness using held-out cities, multiple models and image sampling strategies.

## Data policy

Raw imagery, credentials and large model weights stay on LUMI; do not commit them to Git. An image manifest will reference local paths instead. ACS statistics refer to areas, **not individuals**.

## Status

Initial README / GitHub write-permission test. Model code, dataset schemas, Slurm jobs and analyses have not been committed yet.
