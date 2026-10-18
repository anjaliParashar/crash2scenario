#!/usr/bin/env python3
"""
Cluster scenarios with frequency bias preserved (NO per-combo cap).

Pipeline:
1) Load CSV
2) Prepare categoricals (CP_PreCrash, SV_PreCrash, Work_Zone, Road_Type)
3) Drop redacted / empty narratives
4) Filter dataset by allowed values for cp/sv/work_zone/road_type (optional; defaults = keep all)
5) Sanitize narratives (optional)
6) Sentence embeddings (384) -> PCA(6)
7) One-hot encode 4 categoricals with fixed categories (dim = len(cp)+len(sv)+len(work_zone)+len(road_type))
8) Concatenate -> (cat_dim + 6) and cluster with KMeans
9) Visualize with t-SNE on the final feature space
10) Save clustered CSV + plot + breakdown

Install:
  pip install pandas numpy matplotlib scikit-learn sentence-transformers

Example:
  python3 -m  metadrive.examples.data.cluster_biased \
    --cp "Proceeding Straight" "Making Left Turn" "Making Right Turn" \
    --sv "Proceeding Straight" "Making Left Turn" "Making Right Turn" \
    --work_zone "Yes" "No" \
    --road_type "Highway / Freeway" "Intersection" "Traffic Circle" \
    --k 20 \
    --outdir "out_freq" \
    --random_state 7

If you want to cluster the ENTIRE dataset without filtering, omit --cp/--sv/--work_zone/--road_type
and set --use_all_categories.
"""
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import os
import re
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# ----------------------------
# Redaction filter
# ----------------------------
REDACTED_PATTERN = re.compile(
    r"\[REDACTED, MAY CONTAIN CONFIDENTIAL BUSINESS INFORMATION\]",
    re.IGNORECASE,
)

def drop_redacted_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Narrative" not in df.columns:
        return df
    narrative = df["Narrative"].fillna("").astype(str)
    is_bad = narrative.str.contains(REDACTED_PATTERN, regex=True, na=False) | (narrative.str.strip() == "")

    return df.loc[~is_bad].copy()


# ----------------------------
# Prepare categoricals
# ----------------------------
BASE_COL_MAP = {
    "cp": "CP Pre-Crash Movement",
    "sv": "SV Pre-Crash Movement",
    "road_type": "Roadway Type",
}

def _derive_work_zone_yesno(df: pd.DataFrame) -> pd.Series:
    text_cols = [c for c in ["Narrative", "Roadway Description"] if c in df.columns]
    if not text_cols:
        return pd.Series(["Unknown"] * len(df), index=df.index, name="Work_Zone")

    patt = re.compile(r"\b(work\s*zone|construction|road\s*work|lane\s*closure|flagger)\b", re.IGNORECASE)

    def yesno(i) -> str:
        parts = []
        for c in text_cols:
            v = df.at[i, c]
            if pd.notna(v):
                parts.append(str(v))
        if not parts:
            return "Unknown"
        return "Yes" if patt.search(" ".join(parts)) else "No"

    return pd.Series((yesno(i) for i in df.index), index=df.index, name="Work_Zone")


def prepare_scenario_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CP_PreCrash"] = out[BASE_COL_MAP["cp"]].fillna("Unknown").astype(str) if BASE_COL_MAP["cp"] in out.columns else "Unknown"
    out["SV_PreCrash"] = out[BASE_COL_MAP["sv"]].fillna("Unknown").astype(str) if BASE_COL_MAP["sv"] in out.columns else "Unknown"
    out["Road_Type"] = out[BASE_COL_MAP["road_type"]].fillna("Unknown").astype(str) if BASE_COL_MAP["road_type"] in out.columns else "Unknown"
    out["Work_Zone"] = _derive_work_zone_yesno(out).astype(str)
    return out


# ----------------------------
# Narrative sanitization (optional but recommended)
# ----------------------------
PLACEHOLDER_PATTERN = re.compile(r"\b(x{2,}|\[xxx\]|\[redacted\])\b", re.IGNORECASE)
SPEED_PATTERN = re.compile(r"\b\d+\s?(mph|km/h|kph)\b", re.IGNORECASE)
DATE_TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
ROAD_ID_PATTERN = re.compile(r"\b(i|us|sr|route)\s?-?\s?\d+\b", re.IGNORECASE)

LOCATION_PATTERN = re.compile(
    r"\b(california|new york|texas|arizona|nevada|florida|massachusetts|boston|"
    r"san francisco|los angeles|san diego|oakland|san jose|santa clara|palo alto|mountain view)\b",
    re.IGNORECASE
)
VEHICLE_MAKE_PATTERN = re.compile(
    r"\b(tesla|ford|gm|chevrolet|chevy|toyota|honda|nissan|bmw|audi|"
    r"mercedes|benz|volvo|hyundai|kia|rivian|lucid|subaru|mazda|"
    r"model\s*[sxy3]|cybertruck|f[- ]?150|silverado|camry|corolla|civic|accord|prius)\b",
    re.IGNORECASE
)

def sanitize_narrative(text: str, enabled: bool = True) -> str:
    if not enabled:
        return text if isinstance(text, str) else ""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = PLACEHOLDER_PATTERN.sub(" ", t)
    t = LOCATION_PATTERN.sub(" <LOCATION> ", t)
    t = VEHICLE_MAKE_PATTERN.sub(" <VEHICLE> ", t)
    t = SPEED_PATTERN.sub(" <SPEED> ", t)
    t = DATE_TIME_PATTERN.sub(" <TIME> ", t)
    t = ROAD_ID_PATTERN.sub(" <ROAD_ID> ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ----------------------------
# Embedding -> PCA6
# ----------------------------
def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    E = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return E.astype(np.float32)


def filter_by_lists(
    d: pd.DataFrame,
    cp_list: Optional[List[str]],
    sv_list: Optional[List[str]],
    work_zone_list: Optional[List[str]],
    road_type_list: Optional[List[str]],
) -> pd.DataFrame:
    mask = pd.Series(True, index=d.index)
    if cp_list is not None:
        mask &= d["CP_PreCrash"].isin(cp_list)
    if sv_list is not None:
        mask &= d["SV_PreCrash"].isin(sv_list)
    if work_zone_list is not None:
        mask &= d["Work_Zone"].isin(work_zone_list)
    if road_type_list is not None:
        mask &= d["Road_Type"].isin(road_type_list)
    return d.loc[mask].copy()


def build_features(
    d: pd.DataFrame,
    *,
    cp_list: List[str],
    sv_list: List[str],
    work_zone_list: List[str],
    road_type_list: List[str],
    embedding_model: str,
    random_state: int,
    sanitize: bool,
) -> np.ndarray:
    texts = d["Narrative"].fillna("").astype(str).map(lambda s: sanitize_narrative(s, enabled=sanitize)).tolist()
    E = embed_texts(texts, embedding_model)             # (N, 384)

    pca = PCA(n_components=6, random_state=random_state)
    E6 = pca.fit_transform(E).astype(np.float32)        # (N, 6)

    # Fixed categories ensures stable cat dimension
    fixed_categories = [cp_list, sv_list, work_zone_list, road_type_list]
    cat_cols = ["CP_PreCrash", "SV_PreCrash", "Work_Zone", "Road_Type"]
    C = d[cat_cols].fillna("Unknown").astype(str).values

    ohe = OneHotEncoder(categories=fixed_categories, handle_unknown="ignore", sparse_output=False)
    Ccat = ohe.fit_transform(C).astype(np.float32)      # (N, cat_dim)

    # scale blocks and concat
    X = np.hstack([
        StandardScaler().fit_transform(Ccat),
        StandardScaler().fit_transform(E6),
    ]).astype(np.float32)

    return X


def cluster_and_visualize(
    d: pd.DataFrame,
    X: np.ndarray,
    *,
    k: int,
    outdir: str,
    random_state: int,
    tsne_perplexity: int,
) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)

    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)

    out = d.copy()
    out["cluster"] = labels

    n = len(out)
    perp = min(tsne_perplexity, max(5, (n - 1) // 3)) if n > 10 else max(2, n // 2)

    Z2 = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perp,
        random_state=random_state,
    ).fit_transform(X)

    out["viz_x"] = Z2[:, 0]
    out["viz_y"] = Z2[:, 1]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    for c in sorted(out["cluster"].unique()):
        sub = out[out["cluster"] == c]
        ax.scatter(sub["viz_x"], sub["viz_y"], s=10, alpha=0.75, label=f"cluster {c}")
    ax.set_title(f"KMeans clusters (frequency preserved), k={k}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=1.5)
    plt.tight_layout()

    figpath = os.path.join(outdir, "clusters_tsne.png")
    fig.savefig(figpath, dpi=200)
    plt.close(fig)

    out_csv = os.path.join(outdir, "clustered.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8")

    breakdown = (
        out.groupby(["cluster", "CP_PreCrash", "SV_PreCrash", "Work_Zone", "Road_Type"])
           .size()
           .reset_index(name="count")
           .sort_values(["cluster", "count"], ascending=[True, False])
    )
    breakdown.to_csv(os.path.join(outdir, "cluster_breakdown.csv"), index=False, encoding="utf-8")

    print(f"[Saved] {figpath}")
    print(f"[Saved] {out_csv}")
    print(f"[Saved] {os.path.join(outdir, 'cluster_breakdown.csv')}")
    return out, km



def sample_cluster_centroids(
    df_clustered: pd.DataFrame,
    X: np.ndarray,
    km,
    *,
    n_per_cluster: int = 1,
    ensure_unique: bool = True,
) -> pd.DataFrame:
    """
    Pick scenarios that are closest to each cluster centroid in feature space.

    Parameters
    ----------
    df_clustered : DataFrame
        Must have column 'cluster'. Must be aligned row-wise with X.
    X : np.ndarray
        Feature matrix used for clustering, shape (N, D).
    km : fitted sklearn.cluster.KMeans
        The model used for clustering (has cluster_centers_).
    n_per_cluster : int
        How many representative scenarios to return per cluster (top-k closest).
    ensure_unique : bool
        If True, avoid selecting the same row for multiple clusters (rare but possible
        if clusters overlap / numerical ties). Uses greedy assignment.

    Returns
    -------
    DataFrame
        Representative rows with extra columns:
          - centroid_rank (1..n_per_cluster)
          - dist_to_centroid
    """
    if "cluster" not in df_clustered.columns:
        raise ValueError("df_clustered must contain a 'cluster' column.")
    if X.shape[0] != len(df_clustered):
        raise ValueError("X rows must match df_clustered rows (same order).")

    centers = km.cluster_centers_  # (K, D)
    K = centers.shape[0]

    picked_rows = []
    used_idx = set()

    for c in range(K):
        idxs = np.where(df_clustered["cluster"].to_numpy() == c)[0]
        if len(idxs) == 0:
            continue
        Xc = X[idxs]

        # Euclidean distance to centroid
        dists = np.linalg.norm(Xc - centers[c], axis=1)
        order = np.argsort(dists)

        taken = 0
        for r in order:
            global_i = int(idxs[r])
            if ensure_unique and global_i in used_idx:
                continue
            used_idx.add(global_i)
            row = df_clustered.iloc[global_i].copy()
            row["centroid_rank"] = taken + 1
            row["dist_to_centroid"] = float(dists[r])
            picked_rows.append(row)
            taken += 1
            if taken >= n_per_cluster:
                break

    reps = pd.DataFrame(picked_rows).reset_index(drop=True)
    reps = reps.sort_values(["cluster", "centroid_rank"]).reset_index(drop=True)
    return reps

def sample_nearest_to_point(
    df_clustered: pd.DataFrame,
    X: np.ndarray,
    query: Union[np.ndarray, List[float]],
    *,
    n: int = 10,
    within_cluster: Optional[int] = None,
    return_distances: bool = True,
) -> pd.DataFrame:
    """
    Return the n scenarios closest to a given point in feature space.

    Parameters
    ----------
    df_clustered : DataFrame
        Aligned with X.
    X : np.ndarray
        Feature matrix, (N, D)
    query : array-like
        Query point in same space, shape (D,)
    n : int
        Number of nearest scenarios to return
    within_cluster : int or None
        If set, restrict search to this cluster id.
    return_distances : bool
        If True, add column 'dist_to_query'.

    Returns
    -------
    DataFrame
        Nearest rows sorted by distance.
    """
    if X.shape[0] != len(df_clustered):
        raise ValueError("X rows must match df_clustered rows (same order).")

    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    if q.shape[1] != X.shape[1]:
        raise ValueError(f"query dim {q.shape[1]} must equal X dim {X.shape[1]}")

    if within_cluster is None:
        idxs = np.arange(len(df_clustered))
    else:
        idxs = np.where(df_clustered["cluster"].to_numpy() == within_cluster)[0]
        if len(idxs) == 0:
            return df_clustered.iloc[0:0].copy()

    Xs = X[idxs]
    dists = np.linalg.norm(Xs - q, axis=1)
    order = np.argsort(dists)[: min(n, len(order := np.argsort(dists)))]

    chosen_global = idxs[order]
    out = df_clustered.iloc[chosen_global].copy()
    if return_distances:
        out["dist_to_query"] = dists[order]
    out = out.sort_values("dist_to_query" if return_distances else out.index.name).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--csv", required=True)
    ap.add_argument("--csv", default=None, help="NHTSA ADS crash CSV. Defaults to the copy shipped with this repo.")
    ap.add_argument("--encoding", default="ISO-8859-1")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--outdir", default="out_freq")
    ap.add_argument("--random_state", type=int, default=0, help="KMeans and t-SNE seed")
    ap.add_argument("--pca_random_state", type=int, default=7)
    ap.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--tsne_perplexity", type=int, default=30)
    ap.add_argument("--sanitize", action="store_true", help="Sanitize narratives before embeddings")

    # Optional filtering (if omitted, you can still cluster everything by setting --use_all_categories)
    ap.add_argument("--cp", nargs="+", default=["Proceeding Straight", "Making Left Turn", "Making Right Turn"])
    ap.add_argument("--sv", nargs="+", default=["Proceeding Straight", "Making Left Turn", "Making Right Turn"])
    ap.add_argument("--work_zone", nargs="+", default=["Yes", "No"])
    ap.add_argument("--road_type", nargs="+", default=["Highway / Freeway", "Intersection", "Traffic Circle"])

    ap.add_argument("--use_all_categories", action="store_true",
                    help="Ignore provided category lists and infer categories from data (cat dim not fixed).")

    args = ap.parse_args()

    from crash2scenario.paths import default_crash_csv

    df = pd.read_csv(args.csv or default_crash_csv(), encoding=args.encoding)
    d = prepare_scenario_categoricals(df)
    d = drop_redacted_rows(d)

    # If not using all categories, filter to provided lists
    if not args.use_all_categories:
        if args.cp is None or args.sv is None or args.work_zone is None or args.road_type is None:
            raise SystemExit("Provide --cp --sv --work_zone --road_type OR pass --use_all_categories to cluster all.")
        d = filter_by_lists(d, args.cp, args.sv, args.work_zone, args.road_type)
    if len(d) == 0:
        raise SystemExit("No rows left after redaction removal and filtering.")

    print(f"Rows used for clustering (frequency preserved): {len(d)}")

    # Category lists for fixed one-hot
    if args.use_all_categories:
        cp_list = sorted(d["CP_PreCrash"].dropna().astype(str).unique().tolist())
        sv_list = sorted(d["SV_PreCrash"].dropna().astype(str).unique().tolist())
        wz_list = sorted(d["Work_Zone"].dropna().astype(str).unique().tolist())
        rt_list = sorted(d["Road_Type"].dropna().astype(str).unique().tolist())
        print(f"[info] inferred category sizes: cp={len(cp_list)}, sv={len(sv_list)}, wz={len(wz_list)}, rt={len(rt_list)}")
    else:
        cp_list, sv_list, wz_list, rt_list = args.cp, args.sv, args.work_zone, args.road_type
        print(f"[info] fixed category sizes: cp={len(cp_list)}, sv={len(sv_list)}, wz={len(wz_list)}, rt={len(rt_list)}")

    X = build_features(
        d,
        cp_list=cp_list,
        sv_list=sv_list,
        work_zone_list=wz_list,
        road_type_list=rt_list,
        embedding_model=args.embedding_model,
        random_state=args.pca_random_state,
        sanitize=args.sanitize,
    )

    print(f"Feature dim = {X.shape[1]} (cat_dim + 6)")
    out, km = cluster_and_visualize(
        d,
        X,
        k=args.k,
        outdir=args.outdir,
        random_state=args.random_state,
        tsne_perplexity=args.tsne_perplexity,
    )
    np.save(os.path.join(args.outdir, "features.npy"), X)
    import joblib
    joblib.dump(km, os.path.join(args.outdir, "kmeans.joblib"))

    reps = sample_cluster_centroids(out, X, km, n_per_cluster=1)
    reps.to_csv(os.path.join(args.outdir, "cluster_representatives.csv"), index=False)
    print(reps[["cluster", "centroid_rank", "dist_to_centroid", "CP_PreCrash", "SV_PreCrash", "Work_Zone", "Road_Type"]].head(20))

    for c in range(min(args.k, len(km.cluster_centers_))):
        q = km.cluster_centers_[c]     # 20D point
        near = sample_nearest_to_point(out, X, q, n=1, within_cluster=c)
        print(f"Road: {near['Road_Type'].iloc[0]}, SV:{near['SV_PreCrash'].iloc[0]}, CP:{near['CP_PreCrash'].iloc[0]}, Work Zone:{near['Work_Zone'].iloc[0]} ")


if __name__ == "__main__":
    main()
