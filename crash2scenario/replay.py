"""Replay a released or newly generated scenario in MetaDrive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def scenario_labels(record: Dict[str, Any]) -> List[str]:
    road = str(record["road_type"])
    if "Highway" in road:
        road_label = "Highway_Freeway"
    else:
        road_label = road
    return [road_label, record["sv_precrash"], record["cp_precrash"], record["work_zone"]]


def is_highway(record: Dict[str, Any]) -> bool:
    return "Highway" in str(record.get("road_type", ""))


def load_record(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    required = ("config", "road_type", "sv_precrash", "cp_precrash", "work_zone")
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    cfg = record["config"]
    for key in ("map", "traffic_density", "accident_prob", "steps", "vehicles"):
        if key not in cfg:
            raise ValueError(f"{path} config is missing '{key}'")
    cfg.setdefault("render", {"screen_size": [700, 900], "camera_position": [60, -54]})
    cfg.setdefault("cones", [])
    cfg.setdefault("ego_turn", "straight")
    cfg.setdefault("lead_turn", "straight")
    return record


def replay(record: Dict[str, Any], out_dir: Path, steps: Optional[int] = None) -> Dict[str, Any]:
    """Roll out IDM ego against the scenario. Highway maps use the straight-road runner."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(record["config"]))
    if steps is not None:
        cfg["steps"] = int(steps)
    labels = scenario_labels(record)
    idx = int(record.get("cluster_index", 0))
    revise_iter = 0 if record.get("stage", "initial") == "initial" else 1
    if is_highway(record):
        from crash2scenario.sim_highway import run_old

        return run_old(cfg, labels, idx, revise_iter, out_dir=str(out_dir))
    from crash2scenario.sim_turn import run

    return run(cfg, labels, idx, revise_iter, out_dir=str(out_dir))
