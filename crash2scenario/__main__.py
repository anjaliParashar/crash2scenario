"""Command line for clustering, generation, replay, and release checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crash2scenario.paths import scenarios_dir


def _cmd_cluster(args):
    from crash2scenario.clustering import main as cluster_main

    sys.argv = [
        "crash2scenario.clustering",
        "--csv",
        args.csv,
        "--k",
        str(args.k),
        "--outdir",
        args.outdir,
        "--random_state",
        str(args.random_state),
        "--pca_random_state",
        str(args.pca_random_state),
        "--embedding_model",
        args.embedding_model,
        "--sanitize",
    ]
    cluster_main()


def _cmd_generate(args):
    from crash2scenario.generate import generate_for_cluster

    path = generate_for_cluster(
        Path(args.cluster_dir),
        args.cluster_id,
        Path(args.out_dir),
        n_neighbors=args.neighbors,
        model_name=args.model,
        simulate=args.simulate,
        fine_tune_steps=args.fine_tune_steps,
        sim_steps=args.steps,
    )
    print(path)


def _cmd_replay(args):
    from crash2scenario.replay import load_record, replay

    record = load_record(Path(args.scenario))
    metrics = replay(record, Path(args.out_dir), steps=args.steps)
    print(json.dumps({k: metrics[k] for k in ("min_euclid_dist_m", "any_collision", "map", "gif_path") if k in metrics}, indent=2))


def _cmd_check(args):
    root = Path(args.scenarios)
    errors = []
    records = sorted(root.glob("cluster_*/initial.json"))
    if len(records) < 20:
        errors.append(f"expected 20 released clusters, found {len(records)} under {root}")
    from crash2scenario.replay import load_record

    for path in records:
        try:
            load_record(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
        revised = path.with_name("revised.json")
        if revised.exists():
            try:
                load_record(revised)
            except Exception as exc:
                errors.append(f"{revised}: {exc}")
    videos = list((root / "videos").glob("*.gif")) if (root / "videos").is_dir() else []
    if len(videos) < 20:
        errors.append(f"expected failure-case videos in {root / 'videos'}, found {len(videos)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: {len(records)} scenarios, {len(videos)} videos in {root}")


def main():
    parser = argparse.ArgumentParser(description="crash2scenario paper pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cluster = sub.add_parser("cluster", help="Cluster NHTSA narratives into scenario templates")
    cluster.add_argument("--csv", required=True)
    cluster.add_argument("--k", type=int, default=20)
    cluster.add_argument("--outdir", default="out_freq")
    cluster.add_argument("--random_state", type=int, default=0)
    cluster.add_argument("--pca-random-state", type=int, default=7)
    cluster.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    cluster.set_defaults(func=_cmd_cluster)

    gen = sub.add_parser("generate", help="Paraphrase a cluster and write a MetaDrive config")
    gen.add_argument("--cluster-dir", required=True)
    gen.add_argument("--cluster-id", type=int, required=True)
    gen.add_argument("--out-dir", default="generated")
    gen.add_argument("--neighbors", type=int, default=10)
    gen.add_argument("--model", default="gpt-5.2")
    gen.add_argument("--simulate", action="store_true")
    gen.add_argument("--fine-tune-steps", type=int, default=1)
    gen.add_argument("--steps", type=int, default=None)
    gen.set_defaults(func=_cmd_generate)

    replay = sub.add_parser("replay", help="Simulate a saved scenario JSON")
    replay.add_argument("scenario", help="Path to initial.json or revised.json")
    replay.add_argument("--out-dir", default="rollout")
    replay.add_argument("--steps", type=int, default=None)
    replay.set_defaults(func=_cmd_replay)

    check = sub.add_parser("check", help="Validate the released scenario files")
    check.add_argument("--scenarios", default=str(scenarios_dir()))
    check.set_defaults(func=_cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
