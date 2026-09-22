"""Fast checks for the restorative-beam robustness-study helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


COMMON_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hMSM"
    / "robustness_studies"
    / "common.py"
)
SPEC = importlib.util.spec_from_file_location(
    "matto_robustness_study_common",
    COMMON_PATH,
)
COMMON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMMON
SPEC.loader.exec_module(COMMON)


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        ("mooney", np.exp(2.5 * 0.1 / (1.0 - 1.35 * 0.1))),
        ("guth", 1.0 + 2.5 * 0.1 + 14.1 * 0.1**2),
        ("hill", 1.0 / (1.0 - 2.5 * 0.1)),
    ),
)
def test_g_phi_relations_match_manuscript(model, expected):
    value = COMMON.reinforcement_factor(model, 0.1)
    assert float(value) == pytest.approx(expected)


def test_structured_transfer_preserves_a_bilinear_field():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [100.0, 20.0, 0.0],
        ]
    )
    values = 1.0 + coordinates[:, 0] / 100.0 - coordinates[:, 1] / 40.0
    targets = np.array(
        [
            [25.0, 5.0, 0.0],
            [50.0, 10.0, 0.0],
            [75.0, 15.0, 0.0],
        ]
    )

    transferred = COMMON.interpolate_structured_field(
        coordinates,
        values,
        targets,
    )
    expected = 1.0 + targets[:, 0] / 100.0 - targets[:, 1] / 40.0

    assert transferred == pytest.approx(expected)


def test_angle_transfer_respects_minus_pi_pi_wrap():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [100.0, 20.0, 0.0],
        ]
    )
    angles = np.deg2rad(np.array([179.0, -179.0, 179.0, -179.0]))
    target = np.array([[50.0, 10.0, 0.0]])

    transferred = COMMON.interpolate_angle_field(
        coordinates,
        angles,
        target,
    )

    assert abs(transferred[0]) == pytest.approx(np.pi)


@pytest.mark.parametrize(
    "initialization_id",
    (
        "baseline_uniform",
        "vertical_uniform",
        "underfilled_uniform",
        "spatially_varying",
    ),
)
def test_initializations_are_feasible(initialization_id):
    x, y = np.meshgrid(
        np.linspace(0.0, COMMON.BEAM_LENGTH, 21),
        np.linspace(0.0, COMMON.BEAM_HEIGHT, 5),
    )
    coordinates = np.vstack((x.ravel(), y.ravel()))
    values = COMMON.initialization_values(initialization_id)

    evaluated = {}
    for name, value in values.items():
        if callable(value):
            evaluated[name] = np.asarray(value(coordinates))
        else:
            evaluated[name] = np.full(coordinates.shape[1], value)

    assert np.min(evaluated["rho"]) >= 0.05
    assert np.max(evaluated["rho"]) <= 1.0
    assert np.mean(evaluated["rho"]) <= 0.50 + 1.0e-12
    assert np.min(evaluated["phi"]) >= 0.0
    assert np.max(evaluated["phi"]) <= 0.30
    assert np.mean(evaluated["phi"]) <= 0.10 + 1.0e-12
    assert np.min(evaluated["theta"]) >= -np.pi
    assert np.max(evaluated["theta"]) <= np.pi
