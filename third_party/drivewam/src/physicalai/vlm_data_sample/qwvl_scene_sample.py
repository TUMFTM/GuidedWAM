"""
Two-tier train-split sampler.

Rule:
  * Keep ALL clips with interest_score >= threshold (default 2.0).
  * Random-sample a fraction (default 0.5) of the rest.

Rationale: at 100k / 150k ≈ 67% keep ratio, the goal is "remove the boring
middle", not complex stratified selection. The high-score pool already
contains the corner cases we want every one of; the low-score pool is a
uniform mass of ordinary driving where random subsampling preserves
diversity without wasting data or imposing arbitrary tie-breaks.

Input:  per-clip JSON files produced by qwvl_scene_select_dp.py.
Output: CSV (clip_id, interest_score, tags, video_path, tier).

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd


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
    rows, n_bad = [], 0
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


def sample_train(
    df: pd.DataFrame,
    score_threshold: float,
    low_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Two-tier split: keep all >= threshold, random-sample `low_frac` of the rest."""
    if "interest_score" not in df.columns:
        raise RuntimeError("'interest_score' column missing — did you load tagged JSONs?")

    scores = df["interest_score"].astype(float)
    high = df[scores >= score_threshold].copy()
    low  = df[scores <  score_threshold].copy()

    n_low_keep = int(round(len(low) * low_frac))
    low_sampled = low.sample(n=n_low_keep, random_state=seed) if n_low_keep > 0 else low.iloc[0:0]

    high["tier"] = "high"
    low_sampled["tier"] = "low"
    out = pd.concat([high, low_sampled], ignore_index=True)
    return out


def build_export(sampled: pd.DataFrame, dataset_root: str) -> pd.DataFrame:
    """Narrow to the columns we actually want in the CSV, sorted by score desc."""
    tags_col = sampled.get("tags", pd.Series([[]] * len(sampled), index=sampled.index))
    export = pd.DataFrame({
        "clip_id":        sampled["clip_id"],
        "interest_score": sampled["interest_score"],
        "tags":           tags_col.apply(
            lambda xs: ",".join(xs) if isinstance(xs, list) else ""
        ),
        "video_path":     sampled["clip_id"].apply(lambda c: build_video_path(c, dataset_root)),
        "tier":           sampled["tier"],
    })
    return export.sort_values("interest_score", ascending=False).reset_index(drop=True)


def print_breakdown(df: pd.DataFrame, export: pd.DataFrame, threshold: float) -> None:
    total = len(df)
    n_high = int((df["interest_score"].astype(float) >= threshold).sum())
    n_low  = total - n_high
    kept_high = int((export["tier"] == "high").sum())
    kept_low  = int((export["tier"] == "low").sum())

    print("=" * 60)
    print(f"Corpus                              {total:8d}")
    print(f"  score >= {threshold:<5.2f}                 {n_high:8d}   kept {kept_high}   ({100.0 * kept_high / max(n_high, 1):.1f}%)")
    print(f"  score <  {threshold:<5.2f}                 {n_low:8d}   kept {kept_low}   ({100.0 * kept_low / max(n_low, 1):.1f}%)")
    print(f"Selected total                      {len(export):8d}   ({100.0 * len(export) / max(total, 1):.1f}%)")
    s = export["interest_score"].astype(float)
    print(f"Selected score  mean={s.mean():.2f}  median={s.median():.2f}  "
          f"min={s.min():.2f}  max={s.max():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-tier train-set subsampler from qwvl scene tag JSONs."
    )
    parser.add_argument("--input_dir", nargs="+", required=True,
        help="Directory (or dirs) of per-clip JSON files produced by qwvl_scene_select.py.")
    parser.add_argument("--dataset_root", default="./data/physicalai/front",
        help="Prefix used to build the video_path column in the output CSV.")
    parser.add_argument("--out", default="train_subset.csv",
        help="Output CSV path.")
    parser.add_argument("--score_threshold", type=float, default=2.0,
        help="Clips with interest_score >= this are all kept.")
    parser.add_argument("--low_frac", type=float, default=0.5,
        help="Fraction of clips below the threshold to random-sample (0.0 - 1.0).")
    parser.add_argument("--seed", type=int, default=0,
        help="RNG seed for the low-tier random sample.")
    parser.add_argument("--workers", type=int, default=32,
        help="Threads used to read JSON files in parallel.")
    parser.add_argument("--dry_run", action="store_true",
        help="Print the breakdown but don't write the CSV.")
    args = parser.parse_args()

    if not (0.0 <= args.low_frac <= 1.0):
        raise SystemExit(f"--low_frac must be in [0, 1], got {args.low_frac}")

    input_dirs = [Path(d) for d in args.input_dir]
    for d in input_dirs:
        if not d.is_dir():
            raise FileNotFoundError(d)

    df = load_records(input_dirs, workers=args.workers)
    sampled = sample_train(df, args.score_threshold, args.low_frac, args.seed)
    export = build_export(sampled, args.dataset_root)
    print_breakdown(df, export, args.score_threshold)

    if args.dry_run:
        print("\n[dry_run] CSV not written.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(out_path, index=False)
    print(f"\nTrain subset -> {out_path} ({len(export)} rows)")


if __name__ == "__main__":
    main()
