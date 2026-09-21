"""Optimize the restorative beam on coarse, baseline and fine meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

from common import (
    COMMON_EVALUATION_MESH,
    SUMMARY_FIELDS,
    FrozenDesignEvaluator,
    StudyCase,
    announce,
    run_optimization_case,
    update_case_evaluation,
    write_csv,
)


CASES = {
    "coarse": StudyCase("mesh", "coarse", 100, 20),
    "baseline": StudyCase("mesh", "baseline", 150, 30),
    "fine": StudyCase("mesh", "fine", 200, 40),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", *CASES),
        default="all",
        help="run one mesh case or all three (default: all)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="reuse completed optimizations and only rebuild evaluations/CSVs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "mesh_study",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    comm = MPI.COMM_WORLD
    selected = list(CASES) if args.case == "all" else [args.case]
    summaries = {}
    rows = []

    for case_id in selected:
        case = CASES[case_id]
        output_dir = args.results_dir / case.case_id
        if args.evaluate_only and not (output_dir / "case_summary.json").exists():
            raise FileNotFoundError(
                f"{output_dir} has no completed optimization to evaluate."
            )
        summary = run_optimization_case(case, output_dir, comm)
        summaries[case_id] = summary
        rows.append(summary)
        write_csv(args.results_dir / "summary.csv", rows, SUMMARY_FIELDS, comm)

    evaluator = FrozenDesignEvaluator(
        "mooney", mesh_shape=COMMON_EVALUATION_MESH, comm=comm
    )
    for index, case_id in enumerate(selected):
        case = CASES[case_id]
        output_dir = args.results_dir / case.case_id
        summary = summaries[case_id]
        announce(comm, f"common-mesh evaluation for {case.case_id}")
        evaluator.load(output_dir, case.mesh_shape)
        evaluation = evaluator.evaluate()
        summary = update_case_evaluation(
            output_dir, summary, evaluation, comm
        )
        rows[index] = summary
        write_csv(args.results_dir / "summary.csv", rows, SUMMARY_FIELDS, comm)

    announce(comm, f"mesh study summary: {args.results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
