"""Build one MetaDrive config from a clustered crash neighborhood."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from crash2scenario.clustering import sample_nearest_to_point
from crash2scenario.llm import design_config, fine_tune, narratives_from_frame, paraphrase_narrative
from crash2scenario.replay import replay


def load_cluster_artifacts(cluster_dir: Path):
    cluster_dir = Path(cluster_dir)
    table = pd.read_csv(cluster_dir / "clustered.csv")
    features = np.load(cluster_dir / "features.npy")
    model = joblib.load(cluster_dir / "kmeans.joblib")
    if len(table) != len(features):
        raise ValueError("clustered.csv and features.npy have different numbers of rows.")
    return table, features, model


def generate_for_cluster(
    cluster_dir: Path,
    cluster_id: int,
    out_dir: Path,
    *,
    n_neighbors: int = 10,
    model_name: str = "gpt-5.2",
    simulate: bool = False,
    fine_tune_steps: int = 1,
    sim_steps: Optional[int] = None,
) -> Path:
    table, features, model = load_cluster_artifacts(cluster_dir)
    if cluster_id < 0 or cluster_id >= len(model.cluster_centers_):
        raise ValueError(f"cluster_id {cluster_id} is outside 0..{len(model.cluster_centers_) - 1}")
    query = model.cluster_centers_[cluster_id]
    near = sample_nearest_to_point(table, features, query, n=n_neighbors, within_cluster=cluster_id)
    if near.empty:
        raise RuntimeError(f"Cluster {cluster_id} has no rows.")
    scene = near.iloc[0]
    narrative = paraphrase_narrative(narratives_from_frame(near), scene.to_dict(), model_name=model_name)
    config = design_config(narrative, model_name=model_name)
    record = {
        "cluster_index": int(cluster_id),
        "stage": "initial",
        "road_type": str(scene["Road_Type"]),
        "sv_precrash": str(scene["SV_PreCrash"]),
        "cp_precrash": str(scene["CP_PreCrash"]),
        "work_zone": str(scene["Work_Zone"]),
        "narrative": narrative,
        "source_narratives": narratives_from_frame(near),
        "config": config,
    }
    dest = Path(out_dir) / f"cluster_{cluster_id:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    initial_path = dest / "initial.json"
    initial_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if not simulate:
        return initial_path

    metrics = replay(record, dest / "initial_rollout", steps=sim_steps)
    record["metrics"] = _plain(metrics)
    initial_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    current = record
    for step in range(fine_tune_steps):
        gap = current.get("metrics", {}).get("min_euclid_dist_m")
        collided = bool(current.get("metrics", {}).get("any_collision"))
        if collided or (gap is not None and gap < 2.0):
            break
        tuned = fine_tune(current["config"], current["metrics"], narrative, model_name=model_name)
        revised = dict(current)
        revised["stage"] = "revised"
        revised["config"] = tuned
        revised.pop("metrics", None)
        metrics = replay(revised, dest / f"revised_{step}", steps=sim_steps)
        revised["metrics"] = _plain(metrics)
        (dest / "revised.json").write_text(json.dumps(revised, indent=2), encoding="utf-8")
        current = revised
    return initial_path


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj
