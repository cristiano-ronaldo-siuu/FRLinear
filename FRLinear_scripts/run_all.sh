#!/bin/bash
# Full FRLinear v2 pipeline. Safe to re-run: finished runs are skipped.
#
#   bash FRLinear_scripts/run_all.sh                        # everything
#   bash FRLinear_scripts/run_all.sh --datasets ETTh1,ETTh2 # subset
#   bash FRLinear_scripts/run_all.sh --smoke --datasets ETTh1   # 1-epoch check
#
# Extra arguments are passed to run_all.py (e.g. --max_jobs_small 6 --max_jobs_big 2).
set -e
cd "$(dirname "$0")/.." || exit 1

SUMMARY=./results_summary/
RESULTS=./results_json/
for arg in "$@"; do
  if [ "$arg" == "--smoke" ]; then SUMMARY=./results_summary_smoke/; RESULTS=./results_json_smoke/; fi
done

echo "=== [1/5] regime discovery (train split only) ==="
python FRLinear/data-analysis/src/dataset_analysis/05_generate_regime.py --dataset all

echo "=== [2/5] grid: DLinear, NLinear, FRLinear k x norm ==="
python FRLinear_scripts/run_all.py --stage grid "$@" || echo "some grid runs failed - see logs"

echo "=== [3/5] select (k, norm) on validation ==="
python FRLinear_scripts/select_config.py --results_dir $RESULTS --summary_dir $SUMMARY

echo "=== [4/5] ablation: random grouping, fixed MA, sanity ==="
python FRLinear_scripts/run_all.py --stage ablation "$@" || echo "some ablation runs failed - see logs"

echo "=== [5/5] tables ==="
python FRLinear_scripts/extract_results.py --results_dir $RESULTS --summary_dir $SUMMARY
