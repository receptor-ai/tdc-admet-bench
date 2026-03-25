#!/usr/bin/env python
"""Run Optuna feature selection on a TDC ADMET benchmark."""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tdc_admet_bench.config import suppress_warnings, DEFAULT_FINGERPRINTS
from tdc_admet_bench.optuna_select import run_optuna_feature_selection


def main():
    parser = argparse.ArgumentParser(description="Optuna Feature Selection for TDC ADMET")
    parser.add_argument("--benchmark", default="caco2_wang", help="TDC benchmark name")
    parser.add_argument("--model", default="lgb", help="Model type")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    suppress_warnings()

    model_params = {"n_estimators": 100, "random_state": args.seed, "n_jobs": -1, "verbose": -1}

    result = run_optuna_feature_selection(
        benchmark_name=args.benchmark,
        model_type=args.model,
        model_params=model_params,
        fingerprint_configs=DEFAULT_FINGERPRINTS,
        n_trials=args.n_trials,
        timeout=args.timeout,
        seed=args.seed,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
