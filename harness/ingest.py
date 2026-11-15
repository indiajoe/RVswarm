"""Build the compact per-day store from raw NEID L2 files.

    python -m harness.ingest plan     --n-chunks 200     # write file lists for a SLURM array
    python -m harness.ingest shard    --chunk 17         # one array task
    python -m harness.ingest finalize                    # merge scalars, write manifest
    python -m harness.ingest sample                      # tiny store from the sample files (local)
    python -m harness.ingest toy                         # store of the few frames used by run_tests

The store holds one compressed .npz per observing day (scalars, per-order RVs,
CCFs, activity indices) plus a manifest that maps observations to raw files so
full spectra can be read lazily.
"""
from __future__ import annotations

import argparse
import shutil
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config, l2io


def _find_files(l2_dir) -> list[str]:
    return sorted(str(p) for p in Path(l2_dir).rglob("neidL2_*.fits"))


def _day_of(path) -> int:
    return int(np.floor(l2io.jd_from_filename(path)))     # JD rolls over at 12 UT = pre-dawn at KPNO


def draw_files(by_day: dict, draw: dict) -> dict:
    """Seeded random subset of the raw files. mode 'spectra': n_spectra files drawn uniformly from
    everything. mode 'days': n_days random days, per_day files on each, spread evenly through the day."""
    mode = (draw or {}).get("mode", "none")
    if mode == "none":
        return by_day
    rng = np.random.default_rng(config.derive_seed("draw", draw.get("label", 1)))
    if mode == "spectra":
        files = sorted(f for fs in by_day.values() for f in fs)
        n = min(int(draw["n_spectra"]), len(files))
        pick = [files[i] for i in sorted(rng.choice(len(files), n, replace=False))]
    elif mode == "days":
        k = int(draw["per_day"])
        ok = sorted(d for d, fs in by_day.items() if len(fs) >= k)
        days = sorted(rng.choice(ok, min(int(draw["n_days"]), len(ok)), replace=False))
        pick = []
        for d in days:
            fs = sorted(by_day[int(d)])
            step = len(fs) / k
            off = rng.uniform(0, step)
            pick += [fs[int(off + j * step)] for j in range(k)]
    else:
        sys.exit(f"unknown data.draw.mode: {mode}")
    out = defaultdict(list)
    for f in pick:
        out[_day_of(f)].append(f)
    return dict(out)


def plan(store: Path, l2_dir, n_chunks: int, draw: dict | None = None) -> None:
    files = _find_files(l2_dir)
    if not files:
        sys.exit(f"no neidL2_*.fits under {l2_dir}")
    by_day = defaultdict(list)
    for f in files:
        by_day[_day_of(f)].append(f)
    n_all, d_all = len(files), len(by_day)
    by_day = draw_files(by_day, draw)
    if (store / "days").exists() and any((store / "days").iterdir()):
        old = (store / "plan" / "draw.json")
        if (json.loads(old.read_text()) if old.exists() else {"mode": "none"}) != (draw or {"mode": "none"}):
            sys.exit(f"{store} already holds data from a different draw; use a new data.store_dir for a new draw")
    days = sorted(by_day)
    n_chunks = max(1, min(n_chunks, len(days)))
    (store / "plan").mkdir(parents=True, exist_ok=True)
    for old in (store / "plan").glob("chunk_*.json"):
        old.unlink()
    for c in range(n_chunks):
        chunk = {str(d): by_day[d] for d in days[c::n_chunks]}
        (store / "plan" / f"chunk_{c:05d}.json").write_text(json.dumps(chunk))
    (store / "plan" / "n_chunks").write_text(str(n_chunks))
    (store / "plan" / "draw.json").write_text(json.dumps(draw or {"mode": "none"}))
    n = sum(len(v) for v in by_day.values())
    print(f"{n_all} files on {d_all} days found; {n} files on {len(days)} days selected "
          f"(draw: {json.dumps(draw or {'mode': 'none'})}) -> {n_chunks} chunks in {store / 'plan'}")


def ingest_day(store: Path, day: int, files: list[str], l2_dir) -> int:
    """Every staged file goes in. The data set is curated by the humans: the harness makes
    no quality selection, and a file it cannot use is an error, never a silent drop."""
    out = store / "days" / f"{day}.npz"
    if out.exists():
        return -1
    recs = []
    for f in files:
        try:
            r = l2io.read_lite(f)
        except Exception as e:
            raise SystemExit(f"INGEST STOPPED: cannot read {f}: {e!r}\n"
                             "Fix or remove this file from the curated set and run the ingest again.")
        if not (np.isfinite(r["rv"]) and np.isfinite(r["jd"]) and np.isfinite(r["rv_err"])):
            raise SystemExit(f"INGEST STOPPED: {f} has no finite CCFRVMOD / CCFJDMOD / DVRMSMOD.\n"
                             "Fix or remove this file from the curated set and run the ingest again.")
        r["relpath"] = str(Path(f).resolve().relative_to(Path(l2_dir).resolve()))
        recs.append(r)
    recs.sort(key=lambda r: r["jd"])
    scal_names = sorted(set().union(*(r["scalars"] for r in recs)))
    act_names = recs[0]["act_names"]
    tmp = out.with_suffix(".tmp.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tmp,
        obs_id=np.array([r["obs_id"] for r in recs]),
        relpath=np.array([r["relpath"] for r in recs]),
        jd=np.array([r["jd"] for r in recs]),
        rv=np.array([r["rv"] for r in recs]),
        rv_sum=np.array([r["rv_sum"] for r in recs]),
        rv_err=np.array([r["rv_err"] for r in recs]),
        order_rv=np.array([r["order_rv"] for r in recs]),
        order_wt=np.array([r["order_wt"] for r in recs]),
        ssbz=np.array([r["ssbz"] for r in recs]),
        ccf=np.array([r["ccf"] for r in recs], dtype=np.float32),
        ccf_start=np.array([r["ccf_start"] for r in recs]),
        ccf_step=np.array([r["ccf_step"] for r in recs]),
        scal_names=np.array(scal_names),
        scalars=np.array([[r["scalars"].get(k, np.nan) for k in scal_names] for r in recs]),
        act_names=np.array(act_names),
        act_val=np.array([r["act_val"] if r["act_names"] == act_names else
                          np.full(len(act_names), np.nan) for r in recs]),
        act_unc=np.array([r["act_unc"] if r["act_names"] == act_names else
                          np.full(len(act_names), np.nan) for r in recs]),
    )
    tmp.rename(out)
    return len(recs)


def shard(store: Path, chunk: int, l2_dir) -> None:
    todo = json.loads((store / "plan" / f"chunk_{chunk:05d}.json").read_text())
    for day, files in todo.items():
        n = ingest_day(store, int(day), files, l2_dir)
        print(f"day {day}: {'already done' if n < 0 else f'{n} spectra'}", flush=True)


def finalize(store: Path) -> None:
    """Concatenate the light per-observation columns of all days into index.npz."""
    day_files = sorted((store / "days").glob("[0-9]*.npz"), key=lambda p: int(p.stem))
    if not day_files:
        sys.exit("no day shards found; run the shard step first")
    cols = defaultdict(list)
    scal_names, act_names = None, None
    per_day = []
    for p in day_files:
        z = np.load(p, allow_pickle=False)
        n = len(z["jd"])
        sn, an = list(z["scal_names"]), list(z["act_names"])
        scal_names = scal_names or sn
        act_names = act_names or an
        sc = z["scalars"]
        if sn != scal_names:                                # align columns that differ between days
            m = {k: i for i, k in enumerate(sn)}
            sc = np.stack([sc[:, m[k]] if k in m else np.full(n, np.nan) for k in scal_names], 1)
        av, au = z["act_val"], z["act_unc"]
        if an != act_names:
            m = {k: i for i, k in enumerate(an)}
            av = np.stack([av[:, m[k]] if k in m else np.full(n, np.nan) for k in act_names], 1)
            au = np.stack([au[:, m[k]] if k in m else np.full(n, np.nan) for k in act_names], 1)
        for k in ("obs_id", "relpath", "jd", "rv", "rv_sum", "rv_err", "order_rv", "order_wt",
                  "ssbz", "ccf_start", "ccf_step"):
            cols[k].append(z[k])
        cols["scalars"].append(sc)
        cols["act_val"].append(av)
        cols["act_unc"].append(au)
        cols["day"].append(np.full(n, int(p.stem)))
        cols["row"].append(np.arange(n))
        per_day.append(n)
    merged = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez_compressed(store / "index.tmp.npz", scal_names=np.array(scal_names),
                        act_names=np.array(act_names), **merged)
    (store / "index.tmp.npz").rename(store / "index.npz")
    info = {"n_obs": int(len(merged["jd"])), "n_days": len(day_files),
            "jd_min": float(merged["jd"].min()), "jd_max": float(merged["jd"].max()),
            "median_obs_per_day": float(np.median(per_day))}
    dj = store / "plan" / "draw.json"
    info["draw"] = json.loads(dj.read_text()) if dj.exists() else {"mode": "none"}
    (store / "manifest.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "shard", "finalize", "sample", "toy"])
    ap.add_argument("--n-chunks", type=int, default=200)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--l2-dir")
    ap.add_argument("--store")
    a = ap.parse_args(argv)
    cfg = config.load()
    l2_dir = a.l2_dir or cfg["data"]["l2_dir"]
    if a.cmd in ("sample", "toy"):
        if a.cmd == "toy":                     # the few hand-picked frames for the implementers' test
            l2_dir = a.l2_dir or cfg["data"].get("toy_l2_dir")
            if not l2_dir:
                raise SystemExit("set data.toy_l2_dir in config/harness.yaml (or RVSWARM_TOY_DATA_DIR)")
        store = Path(a.store or cfg["data"]["toy_store_dir" if a.cmd == "toy" else "sample_store_dir"])
        if a.cmd == "toy":
            shutil.rmtree(store, ignore_errors=True)       # always rebuilt from scratch: it is tiny
        plan(store, l2_dir, 1)
        shard(store, 0, l2_dir)
        finalize(store)
        (store / "l2_dir").write_text(str(Path(l2_dir).resolve()))
        return
    store = Path(a.store or cfg["data"]["store_dir"])
    if a.cmd == "plan":
        plan(store, l2_dir, a.n_chunks, cfg["data"].get("draw"))
    elif a.cmd == "shard":
        shard(store, a.chunk, l2_dir)
    else:
        finalize(store)


if __name__ == "__main__":
    main()
