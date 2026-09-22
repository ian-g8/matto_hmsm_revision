"""Shared restorative-beam robustness-study utilities.

The study code intentionally lives beside the hMSM examples.  It mirrors
``examples/hMSM/input_beam.py`` without changing that baseline example or the
general MatTO framework.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, geometry
from dolfinx.mesh import CellType, create_rectangle
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator

from matto import HistoryWriter, OptimizationDriver
from matto.design import volume_constraint
from matto.materials.base import Material
from matto.materials.interpolation import simp
from matto.materials.kinematics import director
from matto.state import set_constant
from matto.utility import Communicator


BEAM_LENGTH = 100.0
BEAM_HEIGHT = 20.0
BASELINE_MESH = (150, 30)
COMMON_EVALUATION_MESH = (200, 40)
G_MODELS = ("mooney", "guth", "hill")
MOVE_LIMIT = 0.05

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

SUMMARY_FIELDS = [
    "study",
    "case_id",
    "mesh_nx",
    "mesh_ny",
    "mesh_size",
    "optimization_model",
    "evaluation_model",
    "initialization_id",
    "objective",
    "common_evaluation_objective",
    "rho_volume",
    "rho_constraint_residual",
    "phi_volume",
    "phi_constraint_residual",
    "common_evaluation_rho_volume",
    "common_evaluation_rho_constraint_residual",
    "common_evaluation_phi_volume",
    "common_evaluation_phi_constraint_residual",
    "iterations",
    "converged",
    "termination_reason",
    "validation_metric",
    "validation_tip_uy",
    "validation_tip_abs_uy",
    "output_dir",
]

EVALUATION_FIELDS = [
    "study",
    "case_id",
    "mesh_nx",
    "mesh_ny",
    "mesh_size",
    "optimization_model",
    "evaluation_model",
    "initialization_id",
    "objective",
    "rho_volume",
    "rho_constraint_residual",
    "phi_volume",
    "phi_constraint_residual",
    "validation_metric",
    "validation_tip_uy",
    "validation_tip_abs_uy",
    "source_output_dir",
]


@dataclass(frozen=True)
class StudyCase:
    """One complete optimization case."""

    study: str
    case_id: str
    mesh_nx: int
    mesh_ny: int
    optimization_model: str = "mooney"
    initialization_id: str = "baseline_uniform"

    @property
    def mesh_shape(self):
        return (self.mesh_nx, self.mesh_ny)

    @property
    def mesh_size(self):
        return max(BEAM_LENGTH / self.mesh_nx, BEAM_HEIGHT / self.mesh_ny)


class StudyHardMagneticSoftMaterial(Material):
    """Study-local hMSM model with a selectable ``G(phi)`` relation."""

    fields = ("rho", "phi", "theta")
    parameters = {
        "G0": None,
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
        "mu0": None,
        "B_rem_mag": None,
        "g_model": "mooney",
        "dim": 2,
    }
    reference = (
        "Galloway & Jha, Model-informed joint material-structural "
        "optimization of hard-magnetic soft materials, arXiv:2607.14397"
    )

    def __init__(self, **parameters):
        parameters = dict(parameters)
        parameters["g_model"] = normalize_g_model(
            parameters.get("g_model", "mooney")
        )
        super().__init__(**parameters)
        if self.dim not in (2, 3):
            raise ValueError(
                "StudyHardMagneticSoftMaterial needs dim=2 or dim=3."
            )
        self.stimuli = {"B_app": (self.dim,)}

    def energy(self, F, fields, stimuli):
        rho_phys = fields["rho"]
        phi_phys = fields["phi"]
        theta_phys = fields["theta"]
        B_app = stimuli["B_app"]

        C = F.T * F
        I1 = ufl.tr(C)
        J = ufl.det(F)

        G_matrix = self.G0 * simp(rho_phys, self.p_rho, self.eps_rho)
        mu = G_matrix * reinforcement_factor(self.g_model, phi_phys)
        K = 500.0 * G_matrix

        W_elastic = (
            (mu / 2.0) * (I1 - 3.0 - 2.0 * ufl.ln(J))
            + (K / 2.0) * (J - 1.0) ** 2
        )

        B_rem = self.B_rem_mag * director(theta_phys, dim=self.dim)
        phi_eff = rho_phys * phi_phys
        W_magnetic = -(1.0 / self.mu0) * phi_eff * ufl.inner(
            F * B_rem, B_app
        )
        return W_elastic + W_magnetic


def normalize_g_model(name):
    """Return a validated lower-case model name."""

    normalized = str(name).strip().lower()
    if normalized not in G_MODELS:
        raise ValueError(
            f"unknown G(phi) model '{name}'; choose one of {G_MODELS}."
        )
    return normalized


def reinforcement_factor(model, phi):
    """Dimensionless ``G(phi) / G_matrix`` used in the manuscript."""

    model = normalize_g_model(model)
    if model == "mooney":
        return ufl.exp(2.5 * phi / (1.0 - 1.35 * phi))
    if model == "guth":
        return 1.0 + 2.5 * phi + 14.1 * phi**2
    return 1.0 / (1.0 - 2.5 * phi)


def initialization_values(initialization_id):
    """Return the three raw-field initial values for one documented seed."""

    if initialization_id == "baseline_uniform":
        return {"rho": 0.50, "phi": 0.10, "theta": 0.0}

    if initialization_id == "low_uniform":
        return {"rho": 0.35, "phi": 0.05, "theta": np.pi / 4.0}

    if initialization_id == "linear_x":
        return {
            "rho": lambda x: 0.30 + 0.40 * x[0] / BEAM_LENGTH,
            "phi": lambda x: 0.05 + 0.10 * x[0] / BEAM_LENGTH,
            "theta": lambda x: -np.pi / 2.0 + np.pi * x[0] / BEAM_LENGTH,
        }

    if initialization_id == "smooth_2d":
        def pattern(x):
            return (
                np.sin(2.0 * np.pi * x[0] / BEAM_LENGTH)
                * np.cos(np.pi * x[1] / BEAM_HEIGHT)
            )

        return {
            "rho": lambda x: (
                0.50
                + 0.15
                * np.cos(2.0 * np.pi * x[0] / BEAM_LENGTH)
                * np.cos(np.pi * x[1] / BEAM_HEIGHT)
            ),
            "phi": lambda x: 0.10 + 0.05 * pattern(x),
            "theta": lambda x: (np.pi / 2.0) * pattern(x),
        }

    raise ValueError(f"unknown initialization '{initialization_id}'.")


def initialization_description(initialization_id):
    """Machine-readable formulas for the initialization metadata."""

    descriptions = {
        "baseline_uniform": {
            "rho": "0.50",
            "phi": "0.10",
            "theta": "0",
        },
        "low_uniform": {
            "rho": "0.35",
            "phi": "0.05",
            "theta": "pi/4",
        },
        "linear_x": {
            "rho": "0.30 + 0.40*x/L",
            "phi": "0.05 + 0.10*x/L",
            "theta": "-pi/2 + pi*x/L",
        },
        "smooth_2d": {
            "rho": "0.50 + 0.15*cos(2*pi*x/L)*cos(pi*y/H)",
            "phi": "0.10 + 0.05*sin(2*pi*x/L)*cos(pi*y/H)",
            "theta": "(pi/2)*sin(2*pi*x/L)*cos(pi*y/H)",
        },
    }
    try:
        return descriptions[initialization_id]
    except KeyError:
        raise ValueError(
            f"unknown initialization '{initialization_id}'."
        ) from None


def build_beam_problem(
    comm,
    *,
    mesh_shape,
    g_model,
    initialization_id,
    output_dir,
    postprocessors=None,
    max_iter=100,
):
    """Build the agreed restorative-beam problem without running it."""

    nx, ny = (int(mesh_shape[0]), int(mesh_shape[1]))
    g_model = normalize_g_model(g_model)
    initial = initialization_values(initialization_id)

    mesh = create_rectangle(
        comm,
        [[0.0, 0.0], [BEAM_LENGTH, BEAM_HEIGHT]],
        [nx, ny],
        cell_type=CellType.quadrilateral,
    )
    if comm.rank == 0:
        mesh_serial = create_rectangle(
            MPI.COMM_SELF,
            [[0.0, 0.0], [BEAM_LENGTH, BEAM_HEIGHT]],
            [nx, ny],
            cell_type=CellType.quadrilateral,
        )
    else:
        mesh_serial = None

    material_parameters = {
        "G0": 100.0,
        "p_rho": 3.0,
        "eps_rho": 1.0e-6,
        "mu0": 1.256e3,
        "B_rem_mag": 200.0,
        "g_model": g_model,
    }

    design_variables = {
        "rho": {
            "active": True,
            "initial": initial["rho"],
            "bounds": (0.05, 1.00),
            "prescribed_value": 1.00,
            "raw_space": ("DG", 0),
            "physical_space": ("CG", 1),
            "operators": [
                {"type": "density_filter", "radius": 1.0},
                {
                    "type": "heaviside",
                    "beta_initial": 1.0,
                    "beta_update_interval": 25,
                    "beta_max": 4.0,
                },
            ],
        },
        "phi": {
            "active": True,
            "initial": initial["phi"],
            "bounds": (0.00, 0.30),
            "prescribed_value": 0.00,
            "raw_space": ("DG", 0),
            "physical_space": ("CG", 1),
            "operators": [{"type": "density_filter", "radius": 1.0}],
        },
        "theta": {
            "active": True,
            "initial": initial["theta"],
            "bounds": (-np.pi, np.pi),
            "prescribed_value": 0.0,
            "raw_space": ("DG", 0),
            "physical_space": ("CG", 1),
            "operators": [{"type": "density_filter", "radius": 1.0}],
        },
    }

    load_cases = [
        {
            "name": "traction_down_B_up",
            "weight": 1.0,
            "body_force": (0.0, 0.0),
            "tractions": {"out_right": (0.0, -0.50)},
            "stimuli": {"B_app": (0.0, 25.0)},
        },
        {
            "name": "traction_up_B_down",
            "weight": 1.0,
            "body_force": (0.0, 0.0),
            "tractions": {"out_right": (0.0, 0.50)},
            "stimuli": {"B_app": (0.0, -25.0)},
        },
    ]

    material = StudyHardMagneticSoftMaterial(**material_parameters)

    def build_objective(u_field, external_work, dx):
        return external_work

    def build_constraints(variables, dx):
        return {
            "rho_volume": volume_constraint(variables["rho"].phys, 0.50, dx),
            "phi_volume": volume_constraint(variables["phi"].phys, 0.10, dx),
        }

    def build_output_fields(variables):
        rho_phys = variables["rho"].phys
        phi_phys = variables["phi"].phys
        theta_phys = variables["theta"].phys
        phi_eff = rho_phys * phi_phys
        m_eff = phi_eff * ufl.as_vector(
            (ufl.cos(theta_phys), ufl.sin(theta_phys))
        )
        return {"phi_eff": phi_eff, "m_eff": m_eff}

    return {
        "mesh": mesh,
        "mesh_serial": mesh_serial,
        "comm": comm,
        "material_parameters": material_parameters,
        "design_variables": design_variables,
        "boundary_conditions": [
            {
                "name": "clamped_left",
                "on_boundary": lambda x: np.isclose(x[0], 0.0),
                "value": (0.0, 0.0),
            },
        ],
        "traction_boundaries": {
            "out_right": lambda x: np.isclose(x[0], BEAM_LENGTH),
        },
        "load_steps": 50,
        "load_cases": load_cases,
        "material": material,
        "build_objective": build_objective,
        "build_constraints": build_constraints,
        "build_output_fields": build_output_fields,
        "requested_output_fields": [
            "u",
            "rho_phys",
            "phi_phys",
            "theta_phys",
            "phi_eff",
            "m_eff",
        ],
        "fem_options": {
            "quadrature_degree": 2,
            "solver_options": {
                "state": {
                    "atol": 1.0e-4,
                    "rtol": 1.0e-4,
                    "max_it": 50,
                    "petsc_options": {
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                },
                "adjoint": {
                    "rtol": 1.0e-8,
                    "atol": 1.0e-12,
                    "petsc_options": {
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                },
                "filter": {
                    "petsc_options": {
                        "ksp_type": "cg",
                        "pc_type": "gamg",
                        "ksp_rtol": 1.0e-10,
                    },
                },
            },
        },
        "optimization_options": {
            "max_iter": int(max_iter),
            "opt_tol": 1.0e-5,
            "move": MOVE_LIMIT,
        },
        "output_options": {
            "output_dir": str(Path(output_dir).resolve()),
            "sim_output_interval": 25,
        },
        "postprocessors": list(postprocessors or []),
    }


def case_settings(case):
    """Serializable record of every setting intentionally held fixed."""

    return {
        "case": asdict(case),
        "beam_dimensions_mm": [BEAM_LENGTH, BEAM_HEIGHT],
        "element_type": "quadrilateral",
        "common_evaluation_mesh": list(COMMON_EVALUATION_MESH),
        "initialization": initialization_description(case.initialization_id),
        "G_phi_model": case.optimization_model,
        "G_phi_relations": {
            "mooney": "Ghat0*exp(2.5*phi/(1-1.35*phi))",
            "guth": "Ghat0*(1+2.5*phi+14.1*phi^2)",
            "hill": "Ghat0/(1-2.5*phi)",
        },
        "material": {
            "strain_energy": "NH2",
            "G0_kPa": 100.0,
            "K_over_G0": 500.0,
            "p_rho": 3.0,
            "eps_rho": 1.0e-6,
            "mu0_mT2_per_kPa": 1.256e3,
            "B_rem_mT": 200.0,
        },
        "design_bounds": {
            "rho": [0.05, 1.0],
            "phi": [0.0, 0.30],
            "theta": [-np.pi, np.pi],
        },
        "volume_upper_bounds": {"rho": 0.50, "phi": 0.10},
        "filter_radii_mm": {"rho": 1.0, "phi": 1.0, "theta": 1.0},
        "density_projection": {
            "beta_initial": 1.0,
            "beta_update_interval": 25,
            "beta_max": 4.0,
        },
        "loads": {
            "optimization": [
                {"traction": [0.0, -0.50], "B_app_mT": [0.0, 25.0]},
                {"traction": [0.0, 0.50], "B_app_mT": [0.0, -25.0]},
            ],
            "validation": {
                "traction": [0.0, -1.00],
                "B_app_y_sweep_mT": list(range(0, 76, 5)),
                "metric": "abs(u_y) at (100, 10) mm and B_app_y=65 mT",
            },
        },
        "load_steps": 50,
        "optimization": {
            "max_iter": 100,
            "opt_tol": 1.0e-5,
            "move": MOVE_LIMIT,
        },
        "output_interval": 25,
    }


def software_metadata(comm):
    """Version and repository information written with every case."""

    if comm.rank != 0:
        return None

    def git(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    try:
        import dolfinx
        dolfinx_version = dolfinx.__version__
    except (ImportError, AttributeError):
        dolfinx_version = "unknown"

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "dolfinx": dolfinx_version,
        "mpi_processes": comm.size,
    }


def announce(comm, message):
    if comm.rank == 0:
        print(f"[robustness-study] {message}", flush=True)


def write_json(path, data, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
    comm.barrier()


def read_json(path, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = None
    return comm.bcast(data, root=0)


def write_csv(path, rows, fieldnames, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
    comm.barrier()


def run_optimization_case(case, output_dir, comm=MPI.COMM_WORLD):
    """Run or reuse one complete optimization and return its summary row."""

    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "case_summary.json"

    if comm.rank == 0:
        complete = summary_path.exists()
        nonempty = output_dir.exists() and any(output_dir.iterdir())
    else:
        complete = None
        nonempty = None
    complete, nonempty = comm.bcast((complete, nonempty), root=0)

    if complete:
        announce(comm, f"reusing completed case {case.case_id}")
        return read_json(summary_path, comm)
    if nonempty:
        raise FileExistsError(
            f"{output_dir} is nonempty but has no case_summary.json. "
            "Move or remove that incomplete directory before retrying."
        )

    announce(
        comm,
        f"starting {case.case_id}: mesh={case.mesh_shape}, "
        f"model={case.optimization_model}, init={case.initialization_id}",
    )
    problem = build_beam_problem(
        comm,
        mesh_shape=case.mesh_shape,
        g_model=case.optimization_model,
        initialization_id=case.initialization_id,
        output_dir=output_dir,
        postprocessors=[HistoryWriter()],
    )
    driver = OptimizationDriver(problem)

    metadata = {
        "status": "running",
        "settings": case_settings(case),
        "software": software_metadata(comm),
    }
    write_json(output_dir / "case_metadata.json", metadata, comm)

    try:
        result = driver.run()
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_json(output_dir / "case_metadata.json", metadata, comm)
        raise

    constraints = result["constraints"]
    converged = bool(
        driver.change <= driver.optimization_tolerance
        and driver.continuation_complete()
    )
    termination_reason = "converged" if converged else "maximum_iterations"

    summary = {
        "study": case.study,
        "case_id": case.case_id,
        "mesh_nx": case.mesh_nx,
        "mesh_ny": case.mesh_ny,
        "mesh_size": case.mesh_size,
        "optimization_model": case.optimization_model,
        "evaluation_model": case.optimization_model,
        "initialization_id": case.initialization_id,
        "objective": float(result["objective"]),
        "common_evaluation_objective": "",
        "rho_volume": float(constraints["rho_volume"]["value"]),
        "rho_constraint_residual": float(
            constraints["rho_volume"]["residual"]
        ),
        "phi_volume": float(constraints["phi_volume"]["value"]),
        "phi_constraint_residual": float(
            constraints["phi_volume"]["residual"]
        ),
        "common_evaluation_rho_volume": "",
        "common_evaluation_rho_constraint_residual": "",
        "common_evaluation_phi_volume": "",
        "common_evaluation_phi_constraint_residual": "",
        "iterations": int(result["iterations"]),
        "converged": converged,
        "termination_reason": termination_reason,
        "validation_metric": "",
        "validation_tip_uy": "",
        "validation_tip_abs_uy": "",
        "output_dir": str(output_dir),
    }

    metadata["status"] = "complete"
    metadata["result"] = summary
    metadata["maximum_displacements"] = {
        name: float(value)
        for name, value in result["max_displacements"].items()
    }
    write_json(output_dir / "case_metadata.json", metadata, comm)
    write_json(summary_path, summary, comm)
    announce(comm, f"completed {case.case_id}")
    return summary


def update_case_evaluation(output_dir, summary, evaluation, comm=MPI.COMM_WORLD):
    """Save one common-mesh evaluation and merge it into the case summary."""

    output_dir = Path(output_dir).resolve()
    write_json(output_dir / "common_evaluation.json", evaluation, comm)
    updated = dict(summary)
    updated["common_evaluation_objective"] = evaluation["objective"]
    updated["common_evaluation_rho_volume"] = evaluation["rho_volume"]
    updated["common_evaluation_rho_constraint_residual"] = evaluation[
        "rho_constraint_residual"
    ]
    updated["common_evaluation_phi_volume"] = evaluation["phi_volume"]
    updated["common_evaluation_phi_constraint_residual"] = evaluation[
        "phi_constraint_residual"
    ]
    updated["validation_metric"] = evaluation["validation_metric"]
    updated["validation_tip_uy"] = evaluation["validation_tip_uy"]
    updated["validation_tip_abs_uy"] = evaluation["validation_tip_abs_uy"]
    write_json(output_dir / "case_summary.json", updated, comm)
    return updated


class FrozenDesignEvaluator:
    """Evaluate saved physical fields on one common mesh without MMA."""

    def __init__(
        self,
        evaluation_model,
        *,
        mesh_shape=COMMON_EVALUATION_MESH,
        comm=MPI.COMM_WORLD,
    ):
        self.comm = comm
        self.mesh_shape = tuple(mesh_shape)
        self.evaluation_model = normalize_g_model(evaluation_model)
        problem = build_beam_problem(
            comm,
            mesh_shape=self.mesh_shape,
            g_model=self.evaluation_model,
            initialization_id="baseline_uniform",
            output_dir=HERE / "results" / ".evaluation_unused",
            postprocessors=[],
            max_iter=1,
        )
        self.driver = OptimizationDriver(problem)
        self._distributors = {
            name: Communicator(variable.phys.function_space, self.driver.mesh_serial)
            for name, variable in self.driver.design_variables.items()
        }

        if comm.rank == 0:
            target_space = fem.functionspace(
                self.driver.mesh_serial, ("CG", 1)
            )
            self.target_coordinates = target_space.tabulate_dof_coordinates()
        else:
            self.target_coordinates = None

    def load(self, source_output_dir, source_mesh_shape):
        """Transfer saved physical fields to the common evaluation mesh."""

        source_output_dir = Path(source_output_dir)
        root_values = None
        error = None
        if self.comm.rank == 0:
            try:
                source_coordinates = serial_cg_coordinates(source_mesh_shape)
                rho = interpolate_structured_field(
                    source_coordinates,
                    np.load(source_output_dir / "final_rho_phys.npy"),
                    self.target_coordinates,
                )
                phi = interpolate_structured_field(
                    source_coordinates,
                    np.load(source_output_dir / "final_phi_phys.npy"),
                    self.target_coordinates,
                )
                theta_source = np.load(
                    source_output_dir / "final_theta_phys.npy"
                )
                theta = interpolate_angle_field(
                    source_coordinates,
                    theta_source,
                    self.target_coordinates,
                )
                root_values = {
                    "rho": np.clip(rho, 0.0, 1.0),
                    "phi": np.clip(phi, 0.0, 0.30),
                    "theta": theta,
                }
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"

        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise RuntimeError(
                f"could not transfer design from {source_output_dir}: {error}"
            )

        for name, variable in self.driver.design_variables.items():
            global_values = self.comm.bcast(
                root_values[name] if self.comm.rank == 0 else None,
                root=0,
            )
            distributor = self._distributors[name]
            index_map = variable.phys.function_space.dofmap.index_map
            block_size = variable.phys.function_space.dofmap.index_map_bs
            owned = index_map.size_local * block_size
            variable.phys.x.array[:owned] = global_values[distributor.idx]
            variable.phys.x.scatter_forward()

    def evaluate(self):
        """Return compliance, constraints and the higher-load tip metric."""

        driver = self.driver
        objective = 0.0
        max_displacements = {}
        for load_case in driver.load_cases:
            name, maximum = driver.solve_load_case(load_case)
            max_displacements[name] = float(maximum)
            value = assemble_global(
                driver.sensitivity.objective_form, self.comm
            )
            objective += float(load_case.get("weight", 1.0)) * value

        constraints = {}
        for name, specification in driver.sensitivity.constraints.items():
            integral = assemble_global(specification["form"], self.comm)
            value = integral / specification["normalization"]
            residual = value / specification["upper_bound"] - 1.0
            constraints[name] = {
                "value": float(value),
                "residual": float(residual),
            }

        validation_sweep = evaluate_validation_field_sweep(
            driver,
            self.comm,
        )
        tip_uy = next(
            item["tip_uy"]
            for item in validation_sweep
            if item["B_app_y_mT"] == 65.0
        )

        return {
            "evaluation_mesh_nx": self.mesh_shape[0],
            "evaluation_mesh_ny": self.mesh_shape[1],
            "evaluation_model": self.evaluation_model,
            "objective": float(objective),
            "rho_volume": constraints["rho_volume"]["value"],
            "rho_constraint_residual": constraints["rho_volume"]["residual"],
            "phi_volume": constraints["phi_volume"]["value"],
            "phi_constraint_residual": constraints["phi_volume"]["residual"],
            "validation_metric": float(abs(tip_uy)),
            "validation_tip_uy": float(tip_uy),
            "validation_tip_abs_uy": float(abs(tip_uy)),
            "validation_field_sweep": validation_sweep,
            "maximum_displacements": max_displacements,
            "transfer": {
                "rho": "bilinear interpolation of final_rho_phys",
                "phi": "bilinear interpolation of final_phi_phys",
                "theta": (
                    "bilinear interpolation of cos(theta) and sin(theta), "
                    "then atan2"
                ),
                "filters_reapplied": False,
            },
        }


def serial_cg_coordinates(mesh_shape):
    """CG1 dof coordinates in the same serial order as final arrays."""

    nx, ny = mesh_shape
    mesh = create_rectangle(
        MPI.COMM_SELF,
        [[0.0, 0.0], [BEAM_LENGTH, BEAM_HEIGHT]],
        [int(nx), int(ny)],
        cell_type=CellType.quadrilateral,
    )
    space = fem.functionspace(mesh, ("CG", 1))
    return space.tabulate_dof_coordinates()


def interpolate_structured_field(source_coordinates, source_values, targets):
    """Bilinearly interpolate a scalar CG1 field between beam meshes."""

    source_coordinates = np.asarray(source_coordinates)
    source_values = np.asarray(source_values, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float)
    if len(source_coordinates) != len(source_values):
        raise ValueError(
            "saved physical field and reconstructed source space have "
            "different sizes."
        )

    x_values = np.unique(np.round(source_coordinates[:, 0], 12))
    y_values = np.unique(np.round(source_coordinates[:, 1], 12))
    if len(x_values) * len(y_values) != len(source_coordinates):
        raise ValueError("source coordinates are not one complete tensor grid.")
    grid = np.empty((len(y_values), len(x_values)), dtype=float)
    x_indices = np.searchsorted(
        x_values, np.round(source_coordinates[:, 0], 12)
    )
    y_indices = np.searchsorted(
        y_values, np.round(source_coordinates[:, 1], 12)
    )
    grid[y_indices, x_indices] = source_values
    interpolator = RegularGridInterpolator(
        (y_values, x_values), grid, method="linear", bounds_error=True
    )
    # CG1 coordinates reconstructed from two independently created meshes can
    # differ from the exact boundary coordinate by roundoff. Round targets to
    # the same precision used to construct the source tensor grid before
    # asking the interpolator to enforce its bounds.
    target_x = np.round(targets[:, 0], 12)
    target_y = np.round(targets[:, 1], 12)
    return np.asarray(
        interpolator(np.column_stack((target_y, target_x))),
        dtype=float,
    )


def interpolate_angle_field(source_coordinates, source_angles, targets):
    """Interpolate a periodic angle without crossing its branch cut."""

    cosine = interpolate_structured_field(
        source_coordinates,
        np.cos(source_angles),
        targets,
    )
    sine = interpolate_structured_field(
        source_coordinates,
        np.sin(source_angles),
        targets,
    )
    magnitude = np.maximum(np.hypot(cosine, sine), 1.0e-14)
    return np.arctan2(sine / magnitude, cosine / magnitude)


def assemble_global(compiled_form, comm):
    local_value = fem.assemble_scalar(compiled_form)
    return float(comm.allreduce(local_value, op=MPI.SUM))


def evaluate_tip_vertical_displacement(mesh, u_field, comm):
    """Evaluate ``u_y`` at the loaded-edge midpoint ``(100, 10)``."""

    point = np.array(
        [[BEAM_LENGTH - 1.0e-10, BEAM_HEIGHT / 2.0, 0.0]],
        dtype=np.float64,
    )
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, point)
    colliding = geometry.compute_colliding_cells(mesh, candidates, point)
    cells = colliding.links(0)

    if len(cells) > 0:
        value = np.asarray(
            u_field.eval(point, np.asarray([cells[0]], dtype=np.int32))
        ).reshape(1, -1)[0, 1]
        local_sum = float(value)
        local_count = 1
    else:
        local_sum = 0.0
        local_count = 0

    total_sum = comm.allreduce(local_sum, op=MPI.SUM)
    total_count = comm.allreduce(local_count, op=MPI.SUM)
    if total_count == 0:
        raise RuntimeError("could not locate the loaded-edge midpoint.")
    return total_sum / total_count


def evaluate_validation_field_sweep(driver, comm):
    """Apply the manuscript's doubled-traction, 0--75 mT field sweep."""

    traction_only = {
        "name": "validation_traction_only",
        "body_force": (0.0, 0.0),
        "tractions": {"out_right": (0.0, -1.00)},
        "stimuli": {"B_app": (0.0, 0.0)},
    }
    driver.solve_load_case(traction_only)

    values = []
    for field_magnitude in range(0, 76, 5):
        if field_magnitude > 0:
            set_constant(
                driver.state.stimuli["B_app"],
                (0.0, float(field_magnitude)),
            )
            driver.state.nonlinear_problem.solve_fem()

        tip_uy = evaluate_tip_vertical_displacement(
            driver.mesh,
            driver.state.u_field,
            comm,
        )
        values.append(
            {
                "B_app_y_mT": float(field_magnitude),
                "tip_uy": float(tip_uy),
                "tip_abs_uy": float(abs(tip_uy)),
            }
        )

    return values


def evaluation_row(case, source_output_dir, result):
    """Flatten a common-mesh result for a long-format CSV."""

    return {
        "study": case.study,
        "case_id": case.case_id,
        "mesh_nx": case.mesh_nx,
        "mesh_ny": case.mesh_ny,
        "mesh_size": case.mesh_size,
        "optimization_model": case.optimization_model,
        "evaluation_model": result["evaluation_model"],
        "initialization_id": case.initialization_id,
        "objective": result["objective"],
        "rho_volume": result["rho_volume"],
        "rho_constraint_residual": result["rho_constraint_residual"],
        "phi_volume": result["phi_volume"],
        "phi_constraint_residual": result["phi_constraint_residual"],
        "validation_metric": result["validation_metric"],
        "validation_tip_uy": result["validation_tip_uy"],
        "validation_tip_abs_uy": result["validation_tip_abs_uy"],
        "source_output_dir": str(Path(source_output_dir).resolve()),
    }
