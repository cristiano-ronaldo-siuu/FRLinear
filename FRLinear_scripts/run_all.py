"""
run_all.py -- parallel experiment runner for FRLinear v2.

Every job is ONE run (one seed) of one configuration, executed as its own
process (run_longExp.py --itr 1 --run_offset r). Finished runs are detected
from their result json, so the script can be stopped and restarted at any time
without repeating work.

Stages
  grid      DLinear, NLinear and the FRLinear selection grid
            (n_regimes in {1,2,3}) x (norm in {none,last}).
  ablation  needs results_summary/selected.json (select_config.py):
            random grouping, fixed decomposition, sanity check, and the
            remaining runs of the selected configuration on big datasets.

Examples (from the repository root)
  python FRLinear_scripts/run_all.py --stage grid --smoke --datasets ETTh1
  python FRLinear_scripts/run_all.py --stage grid
  python FRLinear_scripts/run_all.py --stage ablation
  python FRLinear_scripts/run_all.py --stage grid --dry_run      # list jobs only
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from FRLinear.datasets import DATASETS, PRIORITY, cache_name  # noqa: E402

PY = sys.executable


# ----------------------------------------------------------------------------
# Job construction
# ----------------------------------------------------------------------------
def available_k(ds):
    path = REPO / "FRLinear" / "regime_cache" / f"{cache_name(DATASETS[ds]['file'])}.json"
    with open(path) as f:
        return sorted(int(k) for k in json.load(f)["ward"])


def make_job(ds, pl, run, model, tag, variant, fr=None):
    """tag: unique config name used in model_id; variant: group label in results."""
    return dict(ds=ds, pl=pl, run=run, model=model, tag=tag, variant=variant, fr=fr or {})


def fr_cfg(k, norm, mode="ward", seed=0, decomp="adaptive"):
    return dict(n_regimes=k, norm=norm, regime_mode=mode, rand_seed=seed, decomp=decomp)


def fr_tag(c):
    t = f"FR_k{c['n_regimes']}_{c['norm']}"
    if c["regime_mode"] == "random":
        t += f"_rand{c['rand_seed']}"
    if c["decomp"] == "fixed":
        t += "_fixed"
    return t


def budget(ds, a):
    """Number of runs per job type; big datasets (Electricity, Traffic) use fewer."""
    if DATASETS[ds]["size"] == "big":
        return dict(final=a.runs_big, grid=a.grid_runs_big, fixed=a.abl_runs_big,
                    rand_seeds=list(range(a.rand_seeds_big)), sanity=0)
    return dict(final=a.runs, grid=a.grid_runs, fixed=a.abl_runs,
                rand_seeds=list(range(a.rand_seeds)), sanity=a.sanity_runs)


def horizons(ds, a):
    hs = DATASETS[ds]["horizons"]
    if a.horizons:
        want = {int(h) for h in a.horizons.split(",")}
        hs = [h for h in hs if h in want]
    return hs


def jobs_grid(datasets, a):
    """Baselines (final runs) + FRLinear selection grid (grid runs)."""
    jobs = []
    for ds in datasets:
        b = budget(ds, a)
        ks = [k for k in available_k(ds) if k <= 3]
        for pl in horizons(ds, a):
            for run in range(b["final"]):
                for model in ("DLinear", "NLinear"):
                    jobs.append(make_job(ds, pl, run, model, model, "baseline"))
            for run in range(b["grid"]):
                for k in ks:
                    for norm in ("none", "last"):
                        c = fr_cfg(k, norm)
                        jobs.append(make_job(ds, pl, run, "FRLinear", fr_tag(c), "grid", c))
    return jobs


def jobs_ablation(datasets, a):
    sel_path = Path(a.summary_dir) / "selected.json"
    if not sel_path.exists():
        sys.exit(f"{sel_path} not found - run select_config.py first")
    selected = json.load(open(sel_path))
    jobs = []
    for ds in datasets:
        if ds not in selected:
            print(f"[skip] {ds}: no selected configuration")
            continue
        b = budget(ds, a)
        k_star, norm_star = selected[ds]["n_regimes"], selected[ds]["norm"]
        ks = [k for k in available_k(ds) if 2 <= k <= 3]
        for pl in horizons(ds, a):
            # top-up: the selected configuration gets as many runs as the baselines
            for run in range(b["grid"], b["final"]):
                c = fr_cfg(k_star, norm_star)
                jobs.append(make_job(ds, pl, run, "FRLinear", fr_tag(c), "grid", c))
            # D3: fixed single-kernel (MA-25) decomposition at the selected configuration
            for run in range(b["fixed"]):
                c = fr_cfg(k_star, norm_star, decomp="fixed")
                jobs.append(make_job(ds, pl, run, "FRLinear", fr_tag(c), "fixed_decomp", c))
            # D2: random grouping with the Ward group sizes; partition seed s uses run s
            for k in ks:
                for s in b["rand_seeds"]:
                    c = fr_cfg(k, norm_star, mode="random", seed=s)
                    jobs.append(make_job(ds, pl, s, "FRLinear", fr_tag(c), "random", c))
            # sanity: K=1, fixed MA-25, no norm  ==  DLinear (+ gate)
            for run in range(b["sanity"]):
                c = fr_cfg(1, "none", decomp="fixed")
                jobs.append(make_job(ds, pl, run, "FRLinear", fr_tag(c), "sanity", c))
    return jobs


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
def model_id(j):
    return f"{j['ds']}_{j['tag']}_pl{j['pl']}"


def is_done(j, results_dir):
    pattern = os.path.join(results_dir, f"{model_id(j)}_{j['model']}_*_Exp_{j['run']}.json")
    return len(glob.glob(pattern)) > 0


def command(j, a):
    cfg = DATASETS[j["ds"]]
    pl = j["pl"]
    epochs = 1 if a.smoke else a.train_epochs
    cmd = [PY, "-u", "run_longExp.py",
           "--is_training", "1", "--root_path", a.root_path,
           "--data_path", cfg["file"], "--data", cfg["data"], "--features", "M",
           "--seq_len", str(cfg["seq_len"]), "--pred_len", str(pl),
           "--enc_in", str(cfg["enc_in"]),
           "--batch_size", str(cfg["bs"][pl]), "--learning_rate", str(cfg["lr"][pl]),
           "--train_epochs", str(epochs), "--patience", str(a.patience),
           "--itr", "1", "--run_offset", str(j["run"]), "--seed", str(a.seed),
           "--des", "Exp", "--num_workers", "0",
           "--model", j["model"], "--model_id", model_id(j),
           "--dataset_name", j["ds"], "--variant", j["variant"],
           "--results_dir", a.results_dir,
           "--checkpoints", os.path.join(a.checkpoints, j["ds"]) + "/"]
    for key, val in j["fr"].items():
        cmd += [f"--{key}", str(val)]
    return cmd


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def run_jobs(jobs, a):
    todo = [j for j in jobs if not is_done(j, a.results_dir)]
    print(f"{len(jobs)} jobs in stage, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    per_ds = defaultdict(int)
    for j in todo:
        per_ds[j["ds"]] += 1
    for ds in PRIORITY:
        if per_ds[ds]:
            print(f"   {ds:12s} {per_ds[ds]} jobs")
    if a.dry_run or not todo:
        return []

    day = datetime.now().strftime("%Y-%m-%d")
    log_root = Path(a.log_dir) / day
    log_root.mkdir(parents=True, exist_ok=True)
    runner_log = open(log_root.parent / f"run_all_{a.stage}_{datetime.now():%Y%m%d_%H%M%S}.log", "a")

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    limit = {"small": a.max_jobs_small, "medium": a.max_jobs_medium, "big": a.max_jobs_big}
    running = []                       # (proc, job, start, logfile)
    durations = defaultdict(list)
    failed = []
    done = 0
    t0 = time.time()

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        runner_log.write(line + "\n")
        runner_log.flush()

    def eta():
        rem = 0.0
        remaining = defaultdict(int)
        for j in queue:
            remaining[j["ds"]] += 1
        for p, j, s, _ in running:
            remaining[j["ds"]] += 0.5
        for ds, n in remaining.items():
            d = durations[ds] or [v for vs in durations.values() for v in vs] or [60]
            rem += n * (sum(d) / len(d)) / limit[DATASETS[ds]["size"]]
        return fmt(rem)

    def reap(block):
        nonlocal done
        while True:
            for item in list(running):
                p, j, s, lf = item
                if p.poll() is not None:
                    running.remove(item)
                    lf.close()
                    dur = time.time() - s
                    durations[j["ds"]].append(dur)
                    done += 1
                    ok = p.returncode == 0 and is_done(j, a.results_dir)
                    if not ok:
                        failed.append((j, lf.name))
                    log(f"[{done}/{len(todo)}] {'ok  ' if ok else 'FAIL'} {model_id(j)} run{j['run']} "
                        f"{fmt(dur)} | elapsed {fmt(time.time() - t0)} | ETA ~{eta()}")
            if not block or not running:
                return
            if any(p.poll() is not None for p, *_ in running):
                continue
            time.sleep(1)

    queue = list(todo)
    while queue:
        j = queue[0]
        lim = limit[DATASETS[j["ds"]]["size"]]
        while len(running) >= lim:
            reap(block=False)
            if len(running) >= lim:
                time.sleep(1)
        queue.pop(0)
        lf_path = log_root / j["ds"] / f"{model_id(j)}_run{j['run']}.log"
        lf_path.parent.mkdir(parents=True, exist_ok=True)
        lf = open(lf_path, "w")
        cmd = command(j, a)
        lf.write(f"# {datetime.now().isoformat(timespec='seconds')}\n# {' '.join(cmd)}\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, env=env)
        running.append((p, j, time.time(), lf))
    reap(block=True)

    log(f"stage finished in {fmt(time.time() - t0)}; {len(failed)} failed")
    for j, path in failed:
        log(f"   FAILED {model_id(j)} run{j['run']} -> {path}")
    runner_log.close()
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["grid", "ablation"], required=True)
    p.add_argument("--datasets", default="all", help="comma-separated registry names or 'all'")
    p.add_argument("--root_path", default="./dataset/")
    p.add_argument("--results_dir", default="./results_json/")
    p.add_argument("--log_dir", default="./logs/")
    p.add_argument("--summary_dir", default="./results_summary/")
    p.add_argument("--checkpoints", default="./checkpoints/")
    p.add_argument("--runs", type=int, default=5, help="runs of baselines and of the selected FRLinear config")
    p.add_argument("--grid_runs", type=int, default=3, help="runs per FRLinear grid config (selection)")
    p.add_argument("--abl_runs", type=int, default=3, help="runs of the fixed-MA ablation")
    p.add_argument("--rand_seeds", type=int, default=3, help="random partitions per k (1 run each)")
    p.add_argument("--sanity_runs", type=int, default=1)
    p.add_argument("--runs_big", type=int, default=3, help="big datasets: baselines / selected config")
    p.add_argument("--grid_runs_big", type=int, default=2, help="big datasets: grid runs")
    p.add_argument("--abl_runs_big", type=int, default=2, help="big datasets: fixed-MA runs")
    p.add_argument("--rand_seeds_big", type=int, default=1, help="big datasets: random partitions per k")
    p.add_argument("--horizons", default="", help="optional comma-separated subset of horizons")
    p.add_argument("--seed", type=int, default=2021, help="run r uses seed + r")
    p.add_argument("--train_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max_jobs_small", type=int, default=6)
    p.add_argument("--max_jobs_medium", type=int, default=4)
    p.add_argument("--max_jobs_big", type=int, default=2)
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 1 run, first horizon only, results in *_smoke dirs")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    datasets = PRIORITY if a.datasets == "all" else a.datasets.split(",")
    missing = [d for d in datasets if not (Path(a.root_path) / DATASETS[d]["file"]).exists()]
    if missing:
        print(f"[warn] data file missing, skipped: {missing}")
    datasets = [d for d in datasets if d not in missing]

    if a.smoke:
        a.runs = a.grid_runs = a.abl_runs = a.runs_big = a.grid_runs_big = a.abl_runs_big = 1
        a.rand_seeds = a.rand_seeds_big = a.sanity_runs = 1
        a.results_dir = a.results_dir.rstrip("/") + "_smoke/"
        a.log_dir = a.log_dir.rstrip("/") + "_smoke/"
        a.summary_dir = a.summary_dir.rstrip("/") + "_smoke/"

    jobs = jobs_grid(datasets, a) if a.stage == "grid" else jobs_ablation(datasets, a)
    if a.smoke:
        jobs = [j for j in jobs if j["pl"] == DATASETS[j["ds"]]["horizons"][0] and j["run"] == 0]

    failed = run_jobs(jobs, a)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
