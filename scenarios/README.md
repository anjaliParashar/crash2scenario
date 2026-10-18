# Released scenarios

These are the 20 cluster scenarios from the paper (Table 3 and Table 4), taken from the experiment that produced those minimum-gap numbers.

Each `cluster_XX/` directory contains:

- `initial.json` — paraphrased narrative, MetaDrive config, and logged metrics before fine-tuning
- `revised.json` — config after one fine-tuning step, when the initial rollout was not already a collision

`manifest.json` lists road type, movements, work zone, and minimum gap for each cluster.

`videos/` holds the top-down GIFs recorded during those rollouts. Names look like `R_3_Intersection_..._0.gif` (initial) and `..._1.gif` (revised). A few highway clips from the original run were only a few kilobytes and are omitted. Clusters 4, 11, 12, and 13 were not fine-tuned after the initial collision, matching Table 4.

Replay any file with:

```bash
python -m crash2scenario replay scenarios/cluster_03/initial.json --out-dir rollout
```
