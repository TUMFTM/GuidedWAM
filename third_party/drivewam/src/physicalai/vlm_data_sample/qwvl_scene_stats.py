"""
Analyze per-clip scene tag JSON files produced by qwvl_scene_select.py.

Can be re-run anytime — as more shards finish, stats update. Safe on in-progress runs
because qwvl_scene_select.py writes files atomically (tmp + rename).

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd

BOOL_EVENT_FIELDS = {
    # (prompt_group, field_name)
    ("vru_events", "pedestrian_crossing"),
    ("vru_events", "pedestrian_jaywalking"),
    ("vru_events", "occluded_pedestrian_popout"),
    ("vru_events", "child_or_elderly"),
    ("vru_events", "cyclist_or_ebike_conflict"),
    ("vru_events", "crowd"),
    ("vehicle_events", "cut_in"),
    ("vehicle_events", "cut_out"),
    ("vehicle_events", "sudden_brake_ahead"),
    ("vehicle_events", "wrong_way_vehicle"),
    ("vehicle_events", "large_vehicle_occlusion"),
    ("vehicle_events", "emergency_vehicle"),
    ("vehicle_events", "door_opening"),
    ("vehicle_events", "stopped_or_broken_vehicle"),
    ("intersection_longtail", "unprotected_left"),
    ("intersection_longtail", "roundabout"),
    ("intersection_longtail", "irregular_intersection"),
    ("intersection_longtail", "traffic_police_gesture"),
    ("intersection_longtail", "road_debris"),
    ("intersection_longtail", "accident_scene"),
    ("intersection_longtail", "construction"),
    ("intersection_longtail", "animal_on_road"),
    ("intersection_longtail", "water_puddle"),
    ("intersection_longtail", "railway_crossing"),
}

ENUM_FIELDS = [
    ("scene_attrs", "weather"),
    ("scene_attrs", "lighting"),
    ("scene_attrs", "road_type"),
    ("scene_attrs", "traffic_density"),
    ("scene_attrs", "ego_behavior"),
    ("intersection_longtail", "traffic_light"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_one(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_records(input_dirs: List[Path], workers: int = 32) -> pd.DataFrame:
    paths: List[Path] = []
    for d in input_dirs:
        paths.extend(d.glob("*.json"))
    if not paths:
        raise RuntimeError(f"No JSON files found in {input_dirs}")

    print(f"[load] scanning {len(paths)} JSON files with {workers} threads ...",
          file=sys.stderr)
    rows = []
    n_bad = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rec in enumerate(ex.map(_read_one, paths, chunksize=64), 1):
            if rec is None:
                n_bad += 1
            else:
                rows.append(rec)
            if i % 2000 == 0:
                print(f"[load] {i}/{len(paths)}", file=sys.stderr)
    if n_bad:
        print(f"[warn] {n_bad} unreadable JSON files skipped", file=sys.stderr)
    if not rows:
        raise RuntimeError(f"No records found in {input_dirs}")
    return pd.DataFrame(rows)


def build_video_path(clip_id: str, dataset_root: str) -> str:
    return os.path.join(dataset_root, clip_id, "camera_front_wide_120fov.mp4")


def get_nested(rec: dict, group: str, field: str):
    sub = rec.get(group)
    return sub.get(field) if isinstance(sub, dict) else None


def _scene_attr_series(df: pd.DataFrame, field: str, default: str = "unknown") -> pd.Series:
    """Pull `scene_attrs.<field>` out of each row as a string Series."""
    def pick(sa):
        if not isinstance(sa, dict):
            return default
        v = sa.get(field)
        return str(v) if v is not None else default
    base = df.get("scene_attrs", pd.Series([None] * len(df), index=df.index))
    return base.apply(pick)


# ---------------------------------------------------------------------------
# Stats printers
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    return f"{n:6d}  ({100.0 * n / total:5.1f}%)" if total else f"{n:6d}"


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print("=" * 60)
    print(f"Total tagged clips: {total}")
    if "interest_score" in df.columns:
        s = df["interest_score"].astype(float)
        print(
            f"interest_score:  mean={s.mean():.2f}  median={s.median():.2f}  "
            f"max={s.max():.2f}  min={s.min():.2f}"
        )
    # Count parse failures per sub-block
    err_cols = [g for g, _ in ENUM_FIELDS[:1] + [("scene_attrs", None)]]  # noqa
    for grp in ("scene_attrs", "vru_events", "vehicle_events", "intersection_longtail"):
        n_err = sum(1 for r in df.to_dict("records")
                    if isinstance(r.get(grp), dict) and "_error" in r[grp])
        if n_err:
            print(f"  [!] {grp} parse errors: {n_err}")


def print_score_histogram(df: pd.DataFrame, bins=(0, 2, 4, 6, 8, 10, 15, float("inf"))) -> None:
    if "interest_score" not in df.columns:
        return
    total = len(df)
    s = df["interest_score"].astype(float)
    print("\nScore distribution")
    print("-" * 60)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == float("inf"):
            mask = s >= lo
            label = f"[{lo:5.1f}, inf)"
        else:
            mask = (s >= lo) & (s < hi)
            label = f"[{lo:5.1f}, {hi:5.1f})"
        n = int(mask.sum())
        bar = "#" * max(1, int(40 * n / total)) if n else ""
        print(f"  {label}  {_pct(n, total)}  {bar}")


def print_bool_tag_frequency(df: pd.DataFrame) -> None:
    total = len(df)
    counts = []
    for grp, field in BOOL_EVENT_FIELDS:
        n = sum(1 for r in df.to_dict("records") if get_nested(r, grp, field) is True)
        counts.append((f"{grp[:16]:<16} {field}", n))
    counts.sort(key=lambda x: -x[1])
    print("\nBoolean event frequency (rate across all tagged clips)")
    print("-" * 60)
    for name, n in counts:
        print(f"  {name:<52s} {_pct(n, total)}")


def print_enum_distribution(df: pd.DataFrame) -> None:
    total = len(df)
    for grp, field in ENUM_FIELDS:
        values = [get_nested(r, grp, field) for r in df.to_dict("records")]
        c = Counter(v for v in values if v is not None)
        if not c:
            continue
        print(f"\n{grp}.{field}")
        print("-" * 60)
        for val, n in c.most_common():
            print(f"  {str(val):<30s} {_pct(n, total)}")


# ---------------------------------------------------------------------------
# Ranking / filtering
# ---------------------------------------------------------------------------

def build_ranking_df(df: pd.DataFrame, dataset_root: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "clip_id": df["clip_id"],
        "interest_score": df.get("interest_score", 0.0),
        "tags": df.get("tags", [[] for _ in range(len(df))]).apply(
            lambda xs: ",".join(xs) if isinstance(xs, list) else ""
        ),
        "video_path": df["clip_id"].apply(lambda c: build_video_path(c, dataset_root)),
    })
    return out.sort_values("interest_score", ascending=False).reset_index(drop=True)


def filter_by_tags(
    df: pd.DataFrame,
    any_tags: Optional[List[str]] = None,
    all_tags: Optional[List[str]] = None,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    if min_score is not None and "interest_score" in df.columns:
        mask &= df["interest_score"].astype(float) >= min_score
    tags_col = df.get("tags", pd.Series([[] for _ in range(len(df))], index=df.index))
    if any_tags:
        any_set = set(any_tags)
        mask &= tags_col.apply(lambda ts: bool(any_set.intersection(ts or [])))
    if all_tags:
        all_set = set(all_tags)
        mask &= tags_col.apply(lambda ts: all_set.issubset(set(ts or [])))
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Stratified sampling (rare-tag + bucketed high-score + baseline)
# ---------------------------------------------------------------------------

def _tag_counts(df: pd.DataFrame) -> Counter:
    c: Counter = Counter()
    for ts in df.get("tags", []):
        if isinstance(ts, list):
            for t in ts:
                c[t] += 1
    return c


def stratified_sample(
    df: pd.DataFrame,
    total: int = 1000,
    high_quantile: float = 0.75,
    rare_freq_threshold: float = 0.01,
    rare_per_tag: int = 30,
    baseline_count: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Pick `total` clips via three tiers:

      1. rare-tag safety net: for each tag with frequency < `rare_freq_threshold`,
         take up to `rare_per_tag` top-scoring clips carrying it.
      2. bucketed high-score: from clips with interest_score >= P`high_quantile`,
         partition by (weather, lighting, road_type) and take ~equal quota per
         bucket (top-scoring within each), filling remaining slots.
      3. baseline: random sample from the low-score pool for a calm-scene control.

    Returns the original df restricted to the selected clip_ids, with an extra
    `source` column indicating which tier picked it.
    """
    if len(df) == 0:
        return df.assign(source=pd.Series(dtype=str))

    df = df.copy().reset_index(drop=True)
    df["_tags_list"] = df.get("tags", pd.Series([[]] * len(df))).apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["_weather"]   = _scene_attr_series(df, "weather")
    df["_lighting"]  = _scene_attr_series(df, "lighting")
    df["_road_type"] = _scene_attr_series(df, "road_type")

    selected: dict = {}  # clip_id -> source tier

    # --- Tier 1: rare tags ---
    tag_counts = _tag_counts(df)
    total_n = len(df)
    rare_tags = sorted(
        [t for t, n in tag_counts.items() if 0 < n < rare_freq_threshold * total_n],
        key=lambda t: tag_counts[t],
    )
    for tag in rare_tags:
        subset = df[df["_tags_list"].apply(lambda ts: tag in ts)]
        top = subset.nlargest(min(rare_per_tag, len(subset)), "interest_score")
        for cid in top["clip_id"]:
            selected.setdefault(cid, f"rare:{tag}")

    # --- Tier 2: bucketed high-score ---
    threshold = float(df["interest_score"].quantile(high_quantile)) if "interest_score" in df.columns else 0.0
    remaining = df[~df["clip_id"].isin(selected)]
    high_pool = remaining[remaining["interest_score"] >= threshold].copy()
    high_pool["_bucket"] = (
        high_pool["_weather"] + "|" + high_pool["_lighting"] + "|" + high_pool["_road_type"]
    )

    slots_for_high = max(0, total - len(selected) - baseline_count)
    if slots_for_high > 0 and len(high_pool) > 0:
        n_buckets = max(1, high_pool["_bucket"].nunique())
        per_bucket = max(1, slots_for_high // n_buckets)
        high_selected = (
            high_pool.groupby("_bucket", group_keys=False)
            .apply(lambda g: g.nlargest(min(len(g), per_bucket), "interest_score"))
        )
        if len(high_selected) > slots_for_high:
            high_selected = high_selected.nlargest(slots_for_high, "interest_score")
        elif len(high_selected) < slots_for_high:
            pad = high_pool[~high_pool["clip_id"].isin(high_selected["clip_id"])].nlargest(
                slots_for_high - len(high_selected), "interest_score"
            )
            high_selected = pd.concat([high_selected, pad])
        for cid in high_selected["clip_id"]:
            selected.setdefault(cid, "high_score")

    # --- Tier 3: baseline ---
    remaining = df[~df["clip_id"].isin(selected)]
    low_pool = remaining[remaining["interest_score"] < threshold]
    baseline_n = min(baseline_count, len(low_pool), max(0, total - len(selected)))
    if baseline_n > 0:
        baseline = low_pool.sample(n=baseline_n, random_state=seed)
        for cid in baseline["clip_id"]:
            selected.setdefault(cid, "baseline")

    out = df[df["clip_id"].isin(selected)].drop(
        columns=["_tags_list", "_weather", "_lighting", "_road_type"], errors="ignore"
    ).copy()
    out["source"] = out["clip_id"].map(selected)
    return out.sort_values("interest_score", ascending=False).reset_index(drop=True)


def print_sample_breakdown(sampled: pd.DataFrame, df_all: pd.DataFrame) -> None:
    print("\nStratified sample breakdown")
    print("-" * 60)
    tier = sampled["source"].apply(lambda s: s.split(":")[0] if isinstance(s, str) else "?")
    for name, n in tier.value_counts().items():
        print(f"  {name:<16s} {n:5d}")
    if "interest_score" in sampled.columns:
        s = sampled["interest_score"].astype(float)
        print(f"  score: mean={s.mean():.2f}  median={s.median():.2f}  "
              f"min={s.min():.2f}  max={s.max():.2f}")
    # rare-tag coverage — useful sanity check
    rare_rows = sampled[sampled["source"].str.startswith("rare:", na=False)]
    if len(rare_rows):
        covered = rare_rows["source"].str.replace("rare:", "", regex=False).value_counts()
        print("  rare tags covered:")
        for t, n in covered.items():
            total_in_corpus = sum(1 for ts in df_all.get("tags", [])
                                  if isinstance(ts, list) and t in ts)
            print(f"    {t:<40s} {n:3d} / {total_in_corpus} in corpus")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stats + filtering over scene tag JSONs")
    parser.add_argument("--input_dir", nargs="+", default=[""],
        help="Directory (or dirs) of per-clip JSON files produced by qwvl_scene_select.py.")
    parser.add_argument("--dataset_root", default="./data/physicalai/front",
        help="Used only to write the video_path column in exports.")
    parser.add_argument("--save_csv", default="ranked.csv",
        help="Save full ranking (all clips, sorted by interest_score desc) to this CSV.")
    parser.add_argument("--save_top", default=None,
        help="Save only the top-N interesting clips (after filtering) to this CSV.")
    parser.add_argument("--top_n", type=int, default=200,
        help="How many rows to keep for --save_top and also to print on stdout.")
    parser.add_argument("--min_score", type=float, default=None,
        help="Filter: keep only clips with interest_score >= this.")
    parser.add_argument("--any_tags", nargs="*", default=None,
        help="Filter: keep clips whose tag list intersects with any of these.")
    parser.add_argument("--all_tags", nargs="*", default=None,
        help="Filter: keep only clips whose tag list contains ALL of these.")
    parser.add_argument("--no_print_events", action="store_true",
        help="Suppress per-event frequency and enum breakdowns (keep just summary).")
    parser.add_argument("--workers", type=int, default=32,
        help="Threads used to read JSON files in parallel (I/O bound on NFS).")
    # --- stratified sampling ---
    parser.add_argument("--sample_out", default=None,
        help="If set, write a stratified sample (rare tags + bucketed high-score + baseline) "
             "to this CSV. Columns: clip_id, interest_score, tags, video_path, source.")
    parser.add_argument("--sample_total", type=int, default=1000,
        help="Target size for --sample_out.")
    parser.add_argument("--sample_quantile", type=float, default=0.75,
        help="High-score quantile cutoff (e.g. 0.75 = interest_score >= P75).")
    parser.add_argument("--sample_rare_threshold", type=float, default=0.01,
        help="Tags with frequency < this fraction of all clips are treated as rare.")
    parser.add_argument("--sample_rare_per_tag", type=int, default=30,
        help="Max clips to take per rare tag.")
    parser.add_argument("--sample_baseline", type=int, default=200,
        help="Clips to random-sample from the low-score pool as a calm-scene control.")
    parser.add_argument("--sample_seed", type=int, default=0,
        help="RNG seed for the baseline random sample.")
    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dir]
    for d in input_dirs:
        if not d.is_dir():
            raise FileNotFoundError(d)

    df = load_records(input_dirs, workers=args.workers)

    print_summary(df)
    print_score_histogram(df)
    if not args.no_print_events:
        print_bool_tag_frequency(df)
        print_enum_distribution(df)

    # --- Ranking / filtering exports ---
    ranked = build_ranking_df(df, args.dataset_root)

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out, index=False)
        print(f"\nFull ranking -> {out} ({len(ranked)} rows)")

    # Filter for top clips / top_n
    filtered = filter_by_tags(
        df,
        any_tags=args.any_tags,
        all_tags=args.all_tags,
        min_score=args.min_score,
    )
    ranked_filtered = build_ranking_df(filtered, args.dataset_root).head(args.top_n)

    print(f"\nTop {len(ranked_filtered)} clips", end="")
    filters = []
    if args.min_score is not None:
        filters.append(f"min_score>={args.min_score}")
    if args.any_tags:
        filters.append(f"any_tags={args.any_tags}")
    if args.all_tags:
        filters.append(f"all_tags={args.all_tags}")
    if filters:
        print(f"  (filters: {', '.join(filters)})", end="")
    print()
    print("-" * 60)

    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        preview = ranked_filtered[["clip_id", "interest_score", "tags"]].head(
            min(20, len(ranked_filtered))
        )
        print(preview.to_string(index=False))

    if args.save_top:
        out = Path(args.save_top)
        out.parent.mkdir(parents=True, exist_ok=True)
        ranked_filtered.to_csv(out, index=False)
        print(f"\nTop clips -> {out} ({len(ranked_filtered)} rows)")

    # --- Stratified sampling ---
    if args.sample_out:
        sampled = stratified_sample(
            df,
            total=args.sample_total,
            high_quantile=args.sample_quantile,
            rare_freq_threshold=args.sample_rare_threshold,
            rare_per_tag=args.sample_rare_per_tag,
            baseline_count=args.sample_baseline,
            seed=args.sample_seed,
        )
        export = build_ranking_df(sampled, args.dataset_root)
        source_map = dict(zip(sampled["clip_id"], sampled["source"]))
        export["source"] = export["clip_id"].map(source_map)
        out = Path(args.sample_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        export.to_csv(out, index=False)
        print(f"\nStratified sample -> {out} ({len(export)} rows)")
        print_sample_breakdown(sampled, df)


if __name__ == "__main__":
    main()
