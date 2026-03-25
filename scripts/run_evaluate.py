#!/usr/bin/env python
"""Evaluate a feature combination on TDC ADMET benchmarks."""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tdc_admet_bench.config import suppress_warnings, BENCHMARK_CONFIG, get_ranks
from tdc_admet_bench.evaluate import evaluate_benchmark, evaluate_all_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Evaluate on TDC ADMET benchmarks")
    parser.add_argument("--benchmark", default="all", help="Benchmark name or 'all'")
    parser.add_argument("--model", default="lgb", help="Model type")
    parser.add_argument("--features", type=str, default="ecfp,maccs,desc2D",
                        help="Comma-separated fingerprint kinds")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    suppress_warnings()

    # Parse feature configs from comma-separated kinds
    fp_kinds = [k.strip() for k in args.features.split(",")]
    fingerprint_configs = []
    for kind in fp_kinds:
        if kind in ("ecfp", "fcfp"):
            fingerprint_configs.append({"kind": kind, "radius": 2, "length": 1024})
        elif kind in ("avalon", "rdkit", "topological", "atompair", "pattern", "layered", "secfp"):
            fingerprint_configs.append({"kind": kind, "length": 1024})
        else:
            fingerprint_configs.append({"kind": kind})

    model_params = {"n_estimators": 100, "n_jobs": -1, "verbose": -1}

    if args.benchmark == "all":
        results = evaluate_all_benchmarks(
            model_type=args.model,
            model_params=model_params,
            fingerprint_configs=fingerprint_configs,
            n_seeds=args.seeds,
        )
    else:
        results = evaluate_benchmark(
            args.benchmark,
            model_type=args.model,
            model_params=model_params,
            fingerprint_configs=fingerprint_configs,
            n_seeds=args.seeds,
        )

    # Print ranks
    ranks = get_ranks(results)
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    for name, rank_info in ranks.items():
        if rank_info:
            metric_val = results[name]
            print(f"  {name}: {metric_val[0]:.4f}+-{metric_val[1]:.4f} "
                  f"| Rank: {rank_info['rank']}/{rank_info['total_models']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"results": results, "ranks": {k: v for k, v in ranks.items() if v}}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
