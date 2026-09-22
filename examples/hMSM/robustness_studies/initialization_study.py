"""Test restorative-beam sensitivity to four feasible initial designs."""

from __future__ import annotations

import argparse
from pathlib import Path

import common

from mpi4py import MPI

from common import (
    BASELINE_MESH,
    COMMON_EVALUATION_MESH,
    SUMMARY_FIELDS,
    FrozenDesignEvaluator,
    StudyCase,
    announce,
    run_optimization_case,
    update_case_evaluation,
    write_csv,
)


# This baseline-mesh study uses the same 0.05 move limit as the
# completed G(phi)-model study. The mesh study retains common.MOVE_LIMIT=0.005.
common.MOVE_LIMIT = 0.05

INITIALIZATIONS = (
    "baseline_uniform",
    "vertical_uniform",
    "underfilled_uniform",
    "spatially_varying",
)
CASES = {
    initialization: StudyCase(
        "initialization",
        initialization,
        BASELINE_MESH[0],
        BASELINE_MESH[1],
        initialization_id=initialization,
    )
    for initialization in INITIALIZATIONS
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", *CASES),
        default="all",
        help="run one initialization or all four (default: all)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="reuse completed optimizations and only rebuild evaluations/CSVs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "results"
            / "initialization_study"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    comm = MPI.COMM_WORLD
    selected = list(CASES) if args.case == "all" else [args.case]
    summaries = {}
    rows = []

    for initialization in selected:
        case = CASES[initialization]
        output_dir = args.results_dir / case.case_id
        if args.evaluate_only and not (output_dir / "case_summary.json").exists():
            raise FileNotFoundError(
                f"{output_dir} has no completed optimization to evaluate."
            )
        summary = run_optimization_case(case, output_dir, comm)
        summaries[initialization] = summary
        rows.append(summary)
        write_csv(args.results_dir / "summary.csv", rows, SUMMARY_FIELDS, comm)

    evaluator = FrozenDesignEvaluator(
        "mooney", mesh_shape=COMMON_EVALUATION_MESH, comm=comm
    )
    for index, initialization in enumerate(selected):
        case = CASES[initialization]
        output_dir = args.results_dir / case.case_id
        summary = summaries[initialization]
        announce(comm, f"common-mesh evaluation for {case.case_id}")
        evaluator.load(output_dir, case.mesh_shape)
        evaluation = evaluator.evaluate()
        summary = update_case_evaluation(
            output_dir, summary, evaluation, comm
        )
        rows[index] = summary
        write_csv(args.results_dir / "summary.csv", rows, SUMMARY_FIELDS, comm)

    announce(
        comm,
        f"initialization summary: {args.results_dir / 'summary.csv'}",
    )


if __name__ == "__main__":
    main()
