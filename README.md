# crash2scenario

Code for [Scenario Generation for Testing of Autonomous Driving Systems Using Real-World Failure Records](https://arxiv.org/abs/2606.31131) (NeuS 2026).

The pipeline turns [NHTSA ADS crash records](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting) into MetaDrive scenarios:

1. Keep crashes whose narratives are not redacted and whose pre-crash movements are straight, left, or right, on an intersection, highway, or traffic circle.
2. Cluster those records (one-hot meta variables plus a 6-dimensional MiniLM embedding) into 20 groups.
3. Ask an LLM to paraphrase the nearest narratives into one simulator-feasible narrative, then to emit a MetaDrive config.
4. Roll the config out with an IDM ego vehicle and non-ego policies (`sudden_brake`, `accel_then_brake`, or `idm_nominal`), and optionally fine-tune speeds and gaps from the logged minimum distance.

The 20 scenarios used in the paper, their metrics, and the top-down failure videos are already in [`scenarios/`](scenarios/).

## Setup

Python 3.10 or 3.11. MetaDrive 0.4.3 is the version this code was run with.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Headless machines need an offscreen display for the top-down renderer, for example `export SDL_VIDEODRIVER=dummy`.

Scenario generation calls the OpenAI API. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Replaying the released JSON files does not need a key.

## Replay a released scenario

```bash
python -m crash2scenario replay scenarios/cluster_06/initial.json --out-dir rollout
```

Highway / freeway records use the straight-road runner. Intersections and the traffic circle use the turn runner, which sets NodeNetwork checkpoints for `ego_turn` and `lead_turn`. The GIF and `metrics.json` are written under `--out-dir`.

Check that the release is intact:

```bash
python -m crash2scenario check
python -m unittest tests/test_release.py
```

## Reproduce clustering

```bash
python -m crash2scenario cluster \
  --csv data/Incident_Report_2021.csv \
  --k 20 \
  --outdir out_freq
```

This writes `clustered.csv`, `features.npy`, `kmeans.joblib`, a t-SNE plot, and one representative row per cluster. Defaults match the paper: movements in {Proceeding Straight, Making Left Turn, Making Right Turn}, work zone in {Yes, No}, road type in {Highway / Freeway, Intersection, Traffic Circle}, PCA seed 7, KMeans seed 0, embedding `all-MiniLM-L6-v2`.

## Generate a new scenario

```bash
python -m crash2scenario generate \
  --cluster-dir out_freq \
  --cluster-id 3 \
  --out-dir generated \
  --simulate \
  --fine-tune-steps 1
```

`--simulate` rolls the config out and runs the fine-tuning agent once unless the initial rollout already collides or the minimum gap is under 2 m. Omit it to only write the JSON config.

## Layout

| Path | Role |
| --- | --- |
| `crash2scenario/clustering.py` | Filter, embed, and cluster crash records |
| `crash2scenario/llm.py` | Paraphrase, config generation, and fine-tuning prompts |
| `crash2scenario/sim_turn.py` | Intersection and roundabout rollout, IDM ego, custom non-ego policies |
| `crash2scenario/sim_highway.py` | Highway / freeway rollout |
| `data/Incident_Report_2021.csv` | Public 2021 NHTSA ADS crash table |
| `scenarios/` | Paper scenarios, metrics, and GIFs |

## Citation

```bibtex
@article{parashar2026crash2scenario,
  title={Scenario Generation for Testing of Autonomous Driving Systems Using Real-World Failure Records},
  author={Parashar, Anjali and Fan, Chuchu},
  journal={arXiv preprint arXiv:2606.31131},
  year={2026}
}
```
