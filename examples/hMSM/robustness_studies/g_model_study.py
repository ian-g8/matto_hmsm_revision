"""Optimize and cross-evaluate the restorative beam with three G(phi) models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import common

from mpi4py import MPI

from common import (
    BASELINE_MESH,
    COMMON_EVALUATION_MESH,
    EVALUATION_FIELDS,
    G_MODELS,
    SUMMARY_FIELDS,
    FrozenDesignEvaluator,
    StudyCase,
    announce,
    evaluation_row,
    run_optimization_case,
    update_case_evaluation,
    write_csv,
)


# The G(phi)-model comparison uses a fixed 0.05 MMA move limit.
common.MOVE_LIMIT = 0.05

CASES = {
    model: StudyCase(
        "g_model",
        model,
        BASELINE_MESH[0],
        BASELINE_MESH[1],
        optimization_model=model,
    )
    for model in G_MODELS
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", *CASES),
        default="all",
        help="optimize one model or all three (default: all)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="reuse completed optimizations and rebuild cross-evaluations/CSVs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "g_model_study",
    )
    return parser.parse_args()


def write_cross_table(path, source_models, values, comm):
    if comm.rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["optimization_model", *G_MODELS])
            for source_model in source_models:
                writer.writerow(
                    [source_model]
                    + [values[(source_model, model)] for model in G_MODELS]
                )
    comm.barrier()


def main():
    args = parse_arguments()
    comm = MPI.COMM_WORLD
    source_models = list(CASES) if args.case == "all" else [args.case]
    summaries = {}

    for model in source_models:
        case = CASES[model]
        output_dir = args.results_dir / case.case_id
        if args.evaluate_only and not (output_dir / "case_summary.json").exists():
            raise FileNotFoundError(
                f"{output_dir} has no completed optimization to evaluate."
            )
        summaries[model] = run_optimization_case(case, output_dir, comm)
        write_csv(
            args.results_dir / "summary.csv",
            [summaries[name] for name in source_models if name in summaries],
            SUMMARY_FIELDS,
            comm,
        )

    long_rows = []
    compliance = {}
    diagonal_evaluations = {}
    for evaluation_model in G_MODELS:
        evaluator = FrozenDesignEvaluator(
            evaluation_model,
            mesh_shape=COMMON_EVALUATION_MESH,
            comm=comm,
        )
        for source_model in source_models:
            case = CASES[source_model]
            output_dir = args.results_dir / case.case_id
            announce(
                comm,
                f"cross-evaluating {source_model} design with {evaluation_model}",
            )
            evaluator.load(output_dir, case.mesh_shape)
            result = evaluator.evaluate()
            long_rows.append(evaluation_row(case, output_dir, result))
            compliance[(source_model, evaluation_model)] = result["objective"]
            write_csv(
                args.results_dir / "evaluations_long.csv",
                long_rows,
                EVALUATION_FIELDS,
                comm,
            )
            if source_model == evaluation_model:
                diagonal_evaluations[source_model] = result

    summary_rows = []
    for model in source_models:
        summary = summaries[model]
        if model in diagonal_evaluations:
            summary = update_case_evaluation(
                args.results_dir / model,
                summary,
                diagonal_evaluations[model],
                comm,
            )
        summary_rows.append(summary)

    write_csv(
        args.results_dir / "summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
        comm,
    )
    write_csv(
        args.results_dir / "evaluations_long.csv",
        long_rows,
        EVALUATION_FIELDS,
        comm,
    )
    write_cross_table(
        args.results_dir / "cross_evaluation_compliance.csv",
        source_models,
        compliance,
        comm,
    )
    announce(
        comm,
        "G(phi) cross-evaluation table: "
        f"{args.results_dir / 'cross_evaluation_compliance.csv'}",
    )


if __name__ == "__main__":
    main()
