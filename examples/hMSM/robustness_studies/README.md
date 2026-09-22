# Restorative-beam optimization robustness studies

These scripts reproduce the restorative-beam physics in
`examples/hMSM/input_beam.py` and vary only the quantity named by each study.
They do not modify the baseline example or MatTO's general optimization code.

The common fixed settings are NH2 elasticity, the two equal-and-opposite load
cases, 50 load steps, the original volume constraints and `1.0 mm` filter
radii, density-projection continuation from `beta=1` to `beta=4`, 100 maximum
iterations, and an optimization tolerance of `1e-5`.

The MMA move limit is held fixed within each study: the G(phi)-model and
initialization studies use `0.05`; the mesh study uses `0.005` because
`0.05` produced unstable coarse-mesh updates.

## Installation and launch

From the repository root, install MatTO in the active FEniCSx environment:

```bash
python -m pip install -e .
```

Run a single preliminary baseline case before launching all cases:

```bash
python3 examples/hMSM/robustness_studies/g_model_study.py --case mooney
```

Inspect that case's `history.csv`, `final_results.txt`, final arrays, and `.bp`
datasets. Then run the full studies:

```bash
python3 examples/hMSM/robustness_studies/mesh_study.py
python3 examples/hMSM/robustness_studies/g_model_study.py
python3 examples/hMSM/robustness_studies/initialization_study.py
```

Each script accepts `--case CASE` to run a single case and
`--evaluate-only` to rebuild common-mesh evaluations and CSV files from
completed optimizations. A completed case is reused automatically. A nonempty
case directory without `case_summary.json` is treated as an incomplete run and
is never overwritten automatically.

## Study definitions

### Mesh refinement

`mesh_study.py` runs the complete Mooney optimization with the baseline
initialization on:

| Case | Elements | Element size |
| --- | ---: | ---: |
| coarse | 100 x 20 | 1.0 mm |
| baseline | 150 x 30 | 0.667 mm |
| fine | 200 x 40 | 0.5 mm |

The physical filter radii remain `1.0 mm`. Every finished physical design is
also evaluated on the common `200 x 40` mesh.

### Effective shear-modulus model

`g_model_study.py` runs otherwise identical `150 x 30` optimizations using
Mooney, Guth, and Hill. It evaluates every finished design with every model on
the common `200 x 40` mesh and writes both a long-format CSV and the requested
3 x 3 compliance table.

### Initialization sensitivity

`initialization_study.py` uses Mooney and the `150 x 30` mesh with four
deterministic, feasible raw-field seeds:

| ID | rho | phi | theta |
| --- | --- | --- | --- |
| baseline_uniform | 0.50 | 0.10 | 0 |
| vertical_uniform | 0.50 | 0.10 | pi/2 |
| underfilled_uniform | 0.35 | 0.05 | 0 |
| spatially_varying | varies over x and y, mean 0.50 | varies over x and y, mean 0.10 | varies over x and y |

This is a sensitivity check for a nonconvex problem. Similar results must not
be interpreted as proof of global uniqueness.

## Output

Outputs are placed under `robustness_studies/results/` by default. Every case
has its own directory containing:

- `history.csv`
- `final_results.txt`
- final raw and physical `.npy` arrays for `rho`, `phi`, and `theta`
- one `.bp` dataset per optimization load case
- `case_metadata.json`
- `case_summary.json`
- `common_evaluation.json`

Each study writes `summary.csv`. The model study additionally writes
`evaluations_long.csv` and `cross_evaluation_compliance.csv`.
`validation_metric` is the absolute value of `validation_tip_uy`; both are
retained so downstream tables can use a single scalar without discarding the
displacement direction.

The default layout is:

```text
results/
  mesh_study/{coarse,baseline,fine}/
  g_model_study/{mooney,guth,hill}/
  initialization_study/{baseline_uniform,vertical_uniform,underfilled_uniform,spatially_varying}/
```

The study-level CSV files sit directly inside each study directory. The two
`optimized_design_*.bp` datasets inside every case directory can be opened in
ParaView independently, so no baseline or neighboring case is overwritten.

## Common-mesh transfer and validation

The saved physical fields, rather than the raw fields, are transferred to the
common mesh. `rho` and `phi` use bilinear interpolation. `theta` is transferred
by interpolating `cos(theta)` and `sin(theta)` and reconstructing the angle
with `atan2`, which avoids artificial interpolation across the `-pi/pi` wrap.
Filters and projection are not reapplied on the evaluation mesh.

The validation follows the manuscript load path: the downward `1.00 kPa`
traction is applied first, then held fixed while the upward field is swept from
`0` to `75 mT` in `5 mT` increments. The full tip-displacement sweep is saved
in `common_evaluation.json`. The CSV validation metric is the absolute vertical
displacement at the loaded-edge midpoint, `(100, 10) mm`, at `65 mT`; the signed
value is saved alongside it. It is evaluated from the same frozen physical
design.
