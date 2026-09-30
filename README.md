# No Free Credit — research artifact (RESS submission)

Artifact for the manuscript *"No Free Credit: How Much Simulated Evidence Can Replace Real Events in Zero-Failure
Reliability Demonstration of Perception Systems"* (Lam Quoc Dat, submitted to Reliability Engineering & System Safety).

The artifact reproduces every number, table and figure of the manuscript from the result tables in `data/results/`.
The detector is **frozen — no training is performed anywhere in this study**; all results come from a fixed pre-trained
detector applied to public video benchmarks and to synthetic twins of their events.

## Layout

| Path | Content | Licence |
|---|---|---|
| `code/` | analysis pipeline: corpus rebuild, per-frame hit tables, twin construction, discordance, certificates, audits (`e0_checks.py`, `m3_validity_audit.py`, `m4_exchange_rate.py`, `c2_c5a_empirical.py`, `w0_*.py`, `w2_event_definition.py`, …) | MIT (`LICENSE-CODE`) |
| `tools/` | number/table/figure generators and checkers: `make_numbers.py`, `check_numbers.py`, `make_main_tables.py`, `make_supp_tables.py`, `make_figs.py`, `make_proofs_numbers.py`, `check_front_ress.py` | MIT |
| `data/results/` | result tables used by the manuscript (JSON, CSV, Parquet): event manifests, per-frame hit tables, twin outcomes, certificates, audits | CC BY 4.0 (`LICENSE-DATA`) |
| `data/numbers.tex`, `data/proofs_numbers.tex` | every number printed in the manuscript, one macro each, generated from `data/results/` | CC BY 4.0 |
| `data/sprites/person/` | synthetic person sprites generated with MPFB 2 from its CC0 assets | CC0 (as generated) |
| `figs/` | figures of the manuscript | CC BY 4.0 |
| `manuscript/` | LaTeX sources and PDFs of the manuscript and supplement (added after the data tag); build with `pdflatex` + `bibtex` from `manuscript/` | CC BY 4.0 |

## Reproduce the numbers, tables and figures (no detector, minutes on a laptop)

Python 3.11 with `numpy`, `pandas`, `pyarrow`, `scipy`, `matplotlib`, `opencv-python`:

```bash
python tools/make_numbers.py        # -> manuscript/numbers.tex (one macro per number, source + numerator/denominator)
python tools/check_numbers.py       # regenerates and compares with the committed numbers; 36 independent re-derivations; exit 0
python tools/make_main_tables.py    # -> manuscript/tab_*.tex (grid, must-fail, event definition, C4b, confusion, fidelity, LOSO)
python tools/make_supp_tables.py    # -> manuscript/supp_tables.tex (supplementary tables S1-S13)
python tools/make_figs.py           # -> manuscript/figs/fig_tiers, fig_leak, fig_loso, fig_sprites
python code/w0_exchange_rate_v2.py  # -> figs/fig_exchange_rate (and data/results/exchange_rate_v2.json)
python code/f1_worst_phase_site.py  # -> data/results/worst_phase.json, site_bound.json (worst-phase and site checks)
```

| Manuscript item | Source file(s) in `data/results/` | Generator |
|---|---|---|
| Table 3 (Result 2 credits and two-tier plan) | `c2_mc.json`, `c2_empirical.json` | `make_numbers.py` |
| Result 3–4 numbers, Table 4 (per-camera requirement) | `c3_numbers.json`, `c4_power.json`, `validity_audit.json` | `make_numbers.py`, `make_main_tables.py` |
| Certificates by tier (table + figure) | `e0_units.json` | `make_numbers.py`, `make_figs.py` |
| Exchange rate (table, grid, figure) | `exchange_rate.json`, `exchange_rate_v2.json` | `make_main_tables.py`, `w0_exchange_rate_v2.py` |
| Real vs twin outcomes, LOSO, must-fail | `e0_confusion.json`, `validity_audit.json`, `must_fail_v2.json` | `make_main_tables.py`, `make_figs.py` |
| Event-definition sensitivity | `sens_event_definition.json` | `make_main_tables.py` |
| Tier table and exchange table, worst-phase columns/rows; Table S14–S15 | `worst_phase.json` (from `code/f1_worst_phase_site.py`) | `make_numbers.py`, `make_supp_tables.py` |
| Site-level population bound (Section 6, Tables S16–S17) | `site_bound.json` (from `code/f1_worst_phase_site.py`) | `make_numbers.py`, `make_supp_tables.py` |
| Twin fidelity | `e0_geometry_sensitivity.json`, `twin_discordance_c0_main.json` | `make_main_tables.py` |
| Supplementary tables | all of the above + `e0_event_waterfall.json`, `iw_ablation_v2.json`, `c5a_monotone.json` | `make_supp_tables.py` |

## Re-running the pipeline from raw data (optional, hours)

The raw inputs are **not redistributed**: CDnet 2014, LASIESTA and BMC frames and ground truth (obtain them from their
publishers), the per-frame detection dumps, the detector weights, and the Virtual KITTI 2 vehicle sprites
(CC BY-NC-SA 3.0; cut them with `code/twin/vkitti_sprites.py` from your own copy of Virtual KITTI 2).
Set `THS_DATASETS` to the folder that holds the datasets (default `./datasets`).

* **Detector:** `yolo26s-seg.pt` (Ultralytics), SHA-256 `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4`,
  23,467,933 bytes; loaded only through `code/detector_guard.py`, which refuses any other weights. Settings: confidence
  0.25, IoU 0.7, input 640 px, CPU.
* **Detection dumps:** one JSON per video, `{"frames": [{"frame": i, "boxes": [[x1,y1,x2,y2,score,cls], ...]}]}`, produced
  with the detector and settings above on every frame, frames in text-sorted file order (the per-dataset map to the true
  frame number is in `code/gstar_common.py`). The dumps were produced by a separate batch run and are not part of this
  artifact; their SHA-256 tree hashes (per dataset) are listed in `data/results/corpus17_manifest.json`
  (`raw_detection_dumps`), together with the hashes of the raw ground-truth trees.
* **Corpus:** `python code/corpus17.py` rebuilds the event corpus from the dumps and ground truth.
* **Twins:** `python code/twin/build_twin.py full` (per scene, resumable). **Resource rule:** at most 3 workers; start only
  with at least 4 GB of free RAM and stop the run if free RAM stays below 2 GB. Building the twins of the 78 scenes of the C0 run (76 static + 2 camera-jitter
  scenes) took about 18 CPU-hours on a laptop CPU.
* **Analyses:** `code/e0_checks.py`, `code/m3_validity_audit.py`, `code/m4_exchange_rate.py`, `code/c2_c5a_empirical.py`,
  `code/w0_must_fail_v2.py`, `code/w0_exchange_rate_v2.py`, `code/w2_event_definition.py`, `code/f1_worst_phase_site.py`
  (seed 42 everywhere).

## Citation

See `CITATION.cff`. Release: tag `ress-v1.1` (data + code); the manuscript sources are added in the next commit
and cite the tagged commit hash (`data/results/repo_release.json`).

## Versions

* `ress-v1.1` (2026-09-30) — **domain correction after the F1 checks.** The CDnet 2014 category `cameraJitter` is split
  off the static (main) domain into its own tier, like pan-tilt-zoom, using only CDnet's own category label (a shaking
  camera violates the twin's static-background assumption). Its calibration scenes are `boulevard` and `traffic`
  (`badminton` and `sidewalk` have no empty-background run and were never calibration scenes), so the static domain has
  76 scenes instead of 78. This change was made after the worst-phase and site-level checks of v1.0 had shown the only
  worst-phase false acceptance (`traffic`) in this category; it is a post-hoc change and is disclosed as such in the
  manuscript. No detector or twin was re-run: the jitter tier uses the same C0 twin run. All tier tables, exchange
  rates, leave-one-scene-out, worst-phase and site-level results are recomputed for the new static domain.
* `ress-v1.0` (2026-09-30) — first release (static domain of 78 scenes including `cameraJitter`).
