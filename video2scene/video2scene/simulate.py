"""Run a Phase-2 Genesis simulation from a reviewed Phase-1 Blender handoff."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from loguru import logger
from PIL import Image

from .hashing import sha256_file
from .logging_utils import stage_log
from .simulation_artifacts import (
    encode_video,
    prepare_simulation_output,
    validate_simulation_inputs,
    write_comparison,
    write_contact_sheet,
    write_source_simulation_grid,
)
from .simulation_config import (
    SimulationCommandConfig,
    SimulationRunConfig,
    resolve_simulation_command,
    validate_simulation_options,
)
from .simulation_geometry import (
    horizontal_half_extent,
    quaternion_matrix_wxyz,
    rigid_bottom_z,
    world_bounds,
)
from .simulation_runtime import (
    RuntimeScene,
    as_numpy,
    build_runtime,
    render_frame,
    set_rigid_pose,
    sync_visual_followers,
)
from .spec import write_json


def _env0(value: Any) -> np.ndarray:
    array = as_numpy(value)
    if array.ndim >= 2:
        array = array[0]
    return array


def _quat_error_rad(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    return float(2.0 * math.acos(min(1.0, abs(float(np.dot(actual, expected))))))


def _relax_and_initialize(
    runtime: RuntimeScene, config: SimulationRunConfig
) -> dict[str, Any]:
    dynamic_names = [
        name
        for name, entity in runtime.manifest["resolved_entities"].items()
        if entity["physics_type"] == "rigid"
    ]
    if not runtime.deformable:
        for name in dynamic_names:
            source = runtime.manifest["resolved_entities"][name]
            set_rigid_pose(runtime.physical[name], source)
            velocity = (
                *source["initial_state"]["linear_velocity_m_s"],
                *source["initial_state"]["angular_velocity_rad_s"],
            )
            runtime.physical[name].set_dofs_velocity(velocity)
        sync_visual_followers(runtime)
        return {
            "steps": 0,
            "duration_s": 0.0,
            "method": "exact rigid t0 initialization; no deformable relaxation required",
            "deformables": {},
        }

    for index, name in enumerate(dynamic_names):
        source = runtime.manifest["resolved_entities"][name]
        parked = dict(source)
        parked["position_m"] = [
            source["position_m"][0],
            source["position_m"][1],
            1.8 + 0.25 * index,
        ]
        set_rigid_pose(runtime.physical[name], parked)
        runtime.physical[name].set_dofs_velocity((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    relax_steps = max(0, round(config.relax_s / config.dt))
    for _ in range(relax_steps):
        runtime.scene.step(update_visualizer=False)

    relaxed: dict[str, Any] = {}
    for name, entity in runtime.deformable.items():
        state = entity.get_state()
        positions = as_numpy(state.pos)
        velocities = as_numpy(state.vel)
        if positions.ndim == 3:
            positions = positions[0]
            velocities = velocities[0]
        surface = positions[runtime.surface_indices[name]]
        source_surface = runtime.source_surfaces[name]
        displacement = np.linalg.norm(surface - source_surface, axis=1)
        relaxed[name] = {
            "surface_rms_displacement_m": float(np.sqrt(np.mean(displacement**2))),
            "surface_p95_displacement_m": float(np.quantile(displacement, 0.95)),
            "surface_max_displacement_m": float(displacement.max()),
            "pre_reset_max_speed_m_s": float(np.linalg.norm(velocities, axis=1).max()),
            "velocity_reset_at_t0": True,
            "bounds_m": [surface.min(axis=0).tolist(), surface.max(axis=0).tolist()],
        }
        entity.set_velocity(np.zeros((entity.n_vertices, 3), dtype=np.float64))

    for name in dynamic_names:
        source = runtime.manifest["resolved_entities"][name]
        set_rigid_pose(runtime.physical[name], source)
        velocity = (
            *source["initial_state"]["linear_velocity_m_s"],
            *source["initial_state"]["angular_velocity_rad_s"],
        )
        runtime.physical[name].set_dofs_velocity(velocity)
    sync_visual_followers(runtime)
    return {
        "steps": relax_steps,
        "duration_s": relax_steps * config.dt,
        "method": "gravity seating with dynamic rigid bodies parked, followed by zero soft velocity and exact rigid t0 restore",
        "deformables": relaxed,
    }


def _rigid_sample(runtime: RuntimeScene, name: str) -> dict[str, Any]:
    entity = runtime.physical[name]
    source = runtime.manifest["resolved_entities"][name]
    position = _env0(entity.get_pos()).astype(float)
    quaternion = _env0(entity.get_quat()).astype(float)
    velocity = _env0(entity.get_dofs_velocity()).astype(float)
    return {
        "position_m": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
        "linear_velocity_m_s": velocity[:3].tolist(),
        "angular_velocity_rad_s": velocity[3:6].tolist(),
        "bottom_z_m": rigid_bottom_z(source, position, quaternion),
    }


def _soft_state(runtime: RuntimeScene, name: str) -> tuple[np.ndarray, np.ndarray]:
    state = runtime.deformable[name].get_state()
    positions = as_numpy(state.pos)
    velocities = as_numpy(state.vel)
    if positions.ndim == 3:
        positions = positions[0]
        velocities = velocities[0]
    return positions, velocities


def _soft_surface(runtime: RuntimeScene, name: str) -> tuple[np.ndarray, np.ndarray]:
    positions, velocities = _soft_state(runtime, name)
    indices = runtime.surface_indices[name]
    return positions[indices], velocities[indices]


def _validation_rules(runtime: RuntimeScene) -> list[dict[str, Any]]:
    """Return the explicit scene-specific checks compiled into the handoff."""
    return list(runtime.manifest.get("physics_validation", {}).get("checks", []))


def _make_patch_indices(runtime: RuntimeScene) -> dict[str, dict[str, Any]]:
    patches: dict[str, dict[str, Any]] = {}
    for rule in _validation_rules(runtime):
        if rule["type"] != "soft_impact":
            continue
        body_name = rule["entity"]
        target_name = rule["target"]
        body = runtime.manifest["resolved_entities"][body_name]
        surface, _ = _soft_surface(runtime, target_name)
        xy = np.asarray(body["position_m"][:2], dtype=float)
        if body["collision"]["representation"] == "sphere":
            radius = max(0.065, 1.7 * float(body["collision"]["radius_m"]))
        else:
            radius = max(0.055, 1.8 * max(body["collision"]["dimensions_m"][:2]))
        top_cut = float(surface[:, 2].max() - 0.04)
        selected = np.flatnonzero(
            (np.linalg.norm(surface[:, :2] - xy, axis=1) <= radius)
            & (surface[:, 2] >= top_cut)
        )
        if len(selected) < 16:
            top = np.flatnonzero(surface[:, 2] >= np.quantile(surface[:, 2], 0.7))
            selected = top[
                np.argsort(np.linalg.norm(surface[top, :2] - xy, axis=1))[:64]
            ]
        patches[rule["id"]] = {
            "entity": body_name,
            "target": target_name,
            "surface_indices": selected,
            "radius_m": radius,
            "baseline_top_z_m": float(np.quantile(surface[selected, 2], 0.9)),
        }
    return patches


def _sample_state(
    runtime: RuntimeScene, time_s: float, patches: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "time_s": float(time_s),
        "rigid": {},
        "deformable": {},
        "relations": {},
    }
    soft_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in runtime.deformable:
        positions, velocities = _soft_state(runtime, name)
        indices = runtime.surface_indices[name]
        surface = positions[indices]
        surface_velocities = velocities[indices]
        soft_cache[name] = (surface, surface_velocities)
        tet = positions[runtime.tetrahedra[name]]
        signed_six_volume = np.einsum(
            "ij,ij->i",
            np.cross(tet[:, 1] - tet[:, 0], tet[:, 2] - tet[:, 0]),
            tet[:, 3] - tet[:, 0],
        )
        oriented_volume = signed_six_volume * runtime.reference_tet_signs[name] / 6.0
        record["deformable"][name] = {
            "bounds_m": [surface.min(axis=0).tolist(), surface.max(axis=0).tolist()],
            "center_m": surface.mean(axis=0).tolist(),
            "max_surface_speed_m_s": float(
                np.linalg.norm(surface_velocities, axis=1).max()
            ),
            "minimum_oriented_tet_volume_m3": float(oriented_volume.min()),
            "inverted_tetrahedra": int(np.count_nonzero(oriented_volume <= 0.0)),
        }
    for name, entity in runtime.manifest["resolved_entities"].items():
        if entity["physics_type"] == "rigid":
            record["rigid"][name] = _rigid_sample(runtime, name)
    for rule_id, patch in patches.items():
        target_name = patch["target"]
        surface = soft_cache[target_name][0]
        selected = patch["surface_indices"]
        patch_top = float(np.quantile(surface[selected, 2], 0.9))
        rigid = record["rigid"][patch["entity"]]
        record["relations"][rule_id] = {
            "entity": patch["entity"],
            "target": target_name,
            "patch_top_z_m": patch_top,
            "patch_indentation_m": float(patch["baseline_top_z_m"] - patch_top),
            "vertical_gap_m": float(rigid["bottom_z_m"] - patch_top),
        }
    return record


def _write_render_review(
    config: SimulationRunConfig, validation: dict[str, Any]
) -> None:
    artifact_names = (
        "preview.png",
        "comparison.jpg",
        "contact_sheet.jpg",
        "source_simulation_grid.jpg",
        "simulation.mp4",
    )
    write_json(
        config.sim_dir / "render_review.json",
        {
            "schema_version": "video2scene.genesis-render-review.v1",
            "status": "pending_review_of_this_render",
            "scope": "Phase-2 nominal render after approximate Blender lighting transfer",
            "physics_policy": {
                "initial_state_only": True,
                "pure_simulation_after_t0": True,
                "future_pose_replay": False,
                "nonrigid_replaced_by_rigid": False,
            },
            "lighting_transfer": {
                "model": config.light_model,
                "ambient_light": config.ambient_light,
                "key_light_scale": config.key_light_scale,
                "fill_light_scale": config.fill_light_scale,
                "render_shadows": config.render_shadows,
                "surface_color_space": config.surface_color_space,
                "note": "Genesis Rasterizer approximation of Blender area lights; rendering only, no physics effect.",
            },
            "automated_validation": {
                "passed": validation["passed"],
                "physics_validated": validation["physics_validated"],
            },
            "artifact_sha256": {
                name: sha256_file(config.sim_dir / name)
                for name in artifact_names
                if (config.sim_dir / name).is_file()
            },
        },
    )


def _source_event_window(
    manifest: dict[str, Any], event_name: str
) -> tuple[float, float] | None:
    fps = float(manifest["source"]["fps"])
    for event in manifest["semantics"]["events"]:
        if event["name"] == event_name:
            first, last = event["source_frame_interval"]
            return first / fps, last / fps
    return None


def _record_check(
    checks: dict[str, bool], physics_keys: list[str], name: str, passed: bool
) -> None:
    checks[name] = bool(passed)
    physics_keys.append(name)


def _event_timing(
    manifest: dict[str, Any], event_name: str | None, detected_time: float | None
) -> tuple[list[float] | None, bool | None]:
    if event_name is None:
        return None, None
    source_window = _source_event_window(manifest, event_name)
    if source_window is None or detected_time is None:
        return list(source_window) if source_window else None, False
    tolerance_frames = float(
        manifest.get("physics_validation", {}).get("event_tolerance_frames", 2.0)
    )
    tolerance = tolerance_frames / float(manifest["source"]["fps"])
    valid = (
        source_window[0] - tolerance <= detected_time <= source_window[1] + tolerance
    )
    return list(source_window), bool(valid)


def _validate_soft_impact(
    runtime: RuntimeScene,
    config: SimulationRunConfig,
    records: list[dict[str, Any]],
    rule: dict[str, Any],
    checks: dict[str, bool],
    physics_keys: list[str],
    metrics: dict[str, Any],
) -> None:
    rule_id = rule["id"]
    body_name = rule["entity"]
    target_name = rule["target"]
    relation = [record["relations"][rule_id] for record in records]
    gaps = np.asarray([item["vertical_gap_m"] for item in relation])
    indentation = np.asarray([item["patch_indentation_m"] for item in relation])
    velocities = np.asarray(
        [record["rigid"][body_name]["linear_velocity_m_s"] for record in records]
    )
    contact_threshold = float(rule.get("contact_gap_m", 0.004))
    contact_indices = np.flatnonzero(gaps <= contact_threshold)
    contact_time = (
        float(records[int(contact_indices[0])]["time_s"])
        if len(contact_indices)
        else None
    )
    source_window, timing_ok = _event_timing(
        runtime.manifest, rule.get("event"), contact_time
    )
    response_after_contact = False
    if len(contact_indices):
        start = int(contact_indices[0])
        initial_vertical_speed = max(abs(float(velocities[0, 2])), 1e-6)
        response_after_contact = bool(
            np.max(velocities[start:, 2])
            > float(rule.get("minimum_upward_response_m_s", 0.02))
            or np.min(np.abs(velocities[start:, 2]))
            < float(rule.get("maximum_vertical_speed_fraction", 0.35))
            * initial_vertical_speed
        )

    final = records[-1]
    final_state = final["rigid"][body_name]
    final_speed = float(np.linalg.norm(final_state["linear_velocity_m_s"]))
    position = np.asarray(final_state["position_m"], dtype=float)
    target_bounds = world_bounds(runtime.manifest["resolved_entities"][target_name])
    margin = float(rule.get("final_xy_margin_m", 0.02))
    retained = bool(
        np.all(position[:2] >= target_bounds[0, :2] - margin)
        and np.all(position[:2] <= target_bounds[1, :2] + margin)
    )
    final_gap = float(gaps[-1])
    final_gap_range = rule.get("final_gap_range_m", [-0.08, 0.04])
    supported = bool(final_gap_range[0] <= final_gap <= final_gap_range[1])
    minimum_bottom_z = min(
        record["rigid"][body_name]["bottom_z_m"] for record in records
    )
    floor_name = rule.get("floor_support")
    floor_top = (
        float(world_bounds(runtime.manifest["resolved_entities"][floor_name])[1, 2])
        if floor_name is not None
        else 0.0
    )

    metrics.setdefault("soft_impacts", {})[rule_id] = {
        "entity": body_name,
        "target": target_name,
        "source_event": rule.get("event"),
        "source_window_s": source_window,
        "detected_time_s": contact_time,
        "minimum_gap_m": float(gaps.min()),
        "maximum_patch_indentation_m": float(indentation.max(initial=0.0)),
        "response_after_contact": response_after_contact,
        "minimum_rigid_bottom_z_m": float(minimum_bottom_z),
        "floor_top_z_m": floor_top,
        "final_speed_m_s": final_speed,
        "final_vertical_gap_m": final_gap,
        "final_center_within_target_xy": retained,
        "supported_by_target": supported,
    }
    _record_check(
        checks, physics_keys, f"{rule_id}.contact_detected", contact_time is not None
    )
    if timing_ok is not None:
        _record_check(checks, physics_keys, f"{rule_id}.event_timing", timing_ok)
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.deformable_response",
        float(indentation.max(initial=0.0))
        >= float(rule.get("minimum_indentation_m", 2.0e-4)),
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.rigid_response",
        response_after_contact,
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.no_floor_tunneling",
        minimum_bottom_z
        >= floor_top - float(rule.get("maximum_floor_penetration_m", 0.005)),
    )
    _record_check(checks, physics_keys, f"{rule_id}.retained", retained)
    _record_check(checks, physics_keys, f"{rule_id}.supported", supported)
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.final_speed",
        final_speed <= float(rule.get("maximum_final_speed_m_s", 0.35)),
    )
    if "minimum_final_tilt_deg" in rule:
        quaternion = np.asarray(final_state["quaternion_wxyz"], dtype=float)
        tilt = math.degrees(
            math.acos(min(1.0, abs(float(quaternion_matrix_wxyz(quaternion)[2, 2]))))
        )
        metrics["soft_impacts"][rule_id]["final_tilt_deg"] = tilt
        _record_check(
            checks,
            physics_keys,
            f"{rule_id}.final_tilt",
            tilt >= float(rule["minimum_final_tilt_deg"]),
        )
    if "minimum_final_gap_m" in rule:
        _record_check(
            checks,
            physics_keys,
            f"{rule_id}.final_embedding",
            final_gap >= float(rule["minimum_final_gap_m"]),
        )


def _validate_rolling_transition(
    runtime: RuntimeScene,
    records: list[dict[str, Any]],
    rule: dict[str, Any],
    checks: dict[str, bool],
    physics_keys: list[str],
    metrics: dict[str, Any],
) -> None:
    rule_id = rule["id"]
    body_name = rule["entity"]
    body = runtime.manifest["resolved_entities"][body_name]
    positions = np.asarray(
        [record["rigid"][body_name]["position_m"] for record in records], dtype=float
    )
    linear = np.asarray(
        [record["rigid"][body_name]["linear_velocity_m_s"] for record in records],
        dtype=float,
    )
    angular = np.asarray(
        [record["rigid"][body_name]["angular_velocity_rad_s"] for record in records],
        dtype=float,
    )
    bottoms = np.asarray(
        [record["rigid"][body_name]["bottom_z_m"] for record in records], dtype=float
    )
    times = np.asarray([record["time_s"] for record in records], dtype=float)
    support_bounds = {
        name: world_bounds(runtime.manifest["resolved_entities"][name])
        for name in rule["supports"]
    }
    half_extent = horizontal_half_extent(body)
    gap_min, gap_max = map(float, rule.get("support_gap_range_m", [-0.005, 0.01]))
    overlap_margin = float(rule.get("support_overlap_margin_m", 0.002))
    support_masks = {}
    support_gaps = {}
    for support_name, support_aabb in support_bounds.items():
        overlap = np.all(
            (positions[:, :2] + half_extent + overlap_margin >= support_aabb[0, :2])
            & (positions[:, :2] - half_extent - overlap_margin <= support_aabb[1, :2]),
            axis=1,
        )
        gaps = bottoms - support_aabb[1, 2]
        support_masks[support_name] = overlap & (gaps >= gap_min) & (gaps <= gap_max)
        support_gaps[support_name] = gaps
    supported = np.logical_or.reduce(list(support_masks.values()))
    supported_fraction = float(np.mean(supported))

    target_name = rule["target_support"]
    target_aabb = support_bounds[target_name]
    entry_margin = float(rule.get("target_entry_margin_m", 0.0))
    target_inside = (
        np.all(
            (positions[:, :2] >= target_aabb[0, :2] - entry_margin)
            & (positions[:, :2] <= target_aabb[1, :2] + entry_margin),
            axis=1,
        )
        & (support_gaps[target_name] >= gap_min)
        & (support_gaps[target_name] <= gap_max)
    )
    entry_indices = np.flatnonzero(target_inside)
    entry_time = float(times[entry_indices[0]]) if len(entry_indices) else None
    source_window, timing_ok = _event_timing(
        runtime.manifest, rule.get("event"), entry_time
    )

    axis = int(rule["motion_axis"])
    direction = 1.0 if rule["direction"] == "positive" else -1.0
    displacement = float(direction * (positions[-1, axis] - positions[0, axis]))
    direction_fraction = float(
        np.mean(
            direction * linear[:, axis]
            >= -float(rule.get("velocity_tolerance_m_s", 0.02))
        )
    )
    lateral_axis = 1 - axis
    lateral_drift = float(
        np.max(np.abs(positions[:, lateral_axis] - positions[0, lateral_axis]))
    )
    vertical_excursion = float(np.max(np.abs(positions[:, 2] - positions[0, 2])))
    speeds = np.linalg.norm(linear, axis=1)

    radius = float(body["collision"].get("radius_m", rule.get("radius_m", 0.0)))
    angular_axis = int(rule["angular_axis"])
    rolling_sign = float(rule.get("rolling_sign", 1.0))
    surface_speed = rolling_sign * angular[:, angular_axis] * radius
    moving = np.abs(linear[:, axis]) >= float(
        rule.get("minimum_rolling_speed_m_s", 0.1)
    )
    slip_ratio = np.abs(linear[:, axis] - surface_speed) / np.maximum(
        np.maximum(np.abs(linear[:, axis]), np.abs(surface_speed)), 0.05
    )
    median_slip = float(np.median(slip_ratio[moving])) if np.any(moving) else math.inf
    p95_slip = (
        float(np.quantile(slip_ratio[moving], 0.95)) if np.any(moving) else math.inf
    )
    overlapping_gaps = np.concatenate(
        [
            gaps[
                np.all(
                    (positions[:, :2] + half_extent + overlap_margin >= bounds[0, :2])
                    & (
                        positions[:, :2] - half_extent - overlap_margin <= bounds[1, :2]
                    ),
                    axis=1,
                )
            ]
            for bounds, gaps in (
                (support_bounds[name], support_gaps[name]) for name in rule["supports"]
            )
        ]
    )
    minimum_gap = float(overlapping_gaps.min()) if len(overlapping_gaps) else math.inf

    metrics.setdefault("rolling_transitions", {})[rule_id] = {
        "entity": body_name,
        "supports": list(rule["supports"]),
        "target_support": target_name,
        "source_event": rule.get("event"),
        "source_window_s": source_window,
        "detected_entry_time_s": entry_time,
        "signed_displacement_m": displacement,
        "direction_consistency_fraction": direction_fraction,
        "maximum_lateral_drift_m": lateral_drift,
        "maximum_vertical_excursion_m": vertical_excursion,
        "supported_fraction": supported_fraction,
        "minimum_support_gap_m": minimum_gap,
        "initial_speed_m_s": float(speeds[0]),
        "final_speed_m_s": float(speeds[-1]),
        "maximum_speed_m_s": float(speeds.max()),
        "median_rolling_slip_ratio": median_slip,
        "p95_rolling_slip_ratio": p95_slip,
    }
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.target_entered",
        entry_time is not None,
    )
    if timing_ok is not None:
        _record_check(checks, physics_keys, f"{rule_id}.event_timing", timing_ok)
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.forward_displacement",
        displacement >= float(rule["minimum_displacement_m"]),
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.direction_consistency",
        direction_fraction >= float(rule.get("minimum_direction_fraction", 0.95)),
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.lateral_drift",
        lateral_drift <= float(rule["maximum_lateral_drift_m"]),
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.vertical_excursion",
        vertical_excursion <= float(rule["maximum_vertical_excursion_m"]),
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.support_continuity",
        supported_fraction >= float(rule["minimum_supported_fraction"]),
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.no_support_tunneling",
        minimum_gap >= gap_min,
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.rolling_consistency",
        median_slip <= float(rule["maximum_median_slip_ratio"]),
    )
    minimum_final_speed = (
        float(speeds[0]) * float(rule["minimum_final_speed_ratio"])
        if "minimum_final_speed_ratio" in rule
        else float(rule["minimum_final_speed_m_s"])
    )
    metrics["rolling_transitions"][rule_id]["minimum_final_speed_m_s"] = (
        minimum_final_speed
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.final_speed",
        float(speeds[-1]) >= minimum_final_speed,
    )
    _record_check(
        checks,
        physics_keys,
        f"{rule_id}.no_unphysical_speed_gain",
        float(speeds.max())
        <= max(float(speeds[0]), 1e-6)
        * float(rule.get("maximum_speed_gain_ratio", 1.2)),
    )
    if rule.get("require_final_on_target", True):
        _record_check(
            checks,
            physics_keys,
            f"{rule_id}.final_on_target",
            bool(target_inside[-1]),
        )


def _validate_run(
    runtime: RuntimeScene,
    config: SimulationRunConfig,
    relaxation: dict[str, Any],
    records: list[dict[str, Any]],
    video_probe: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    physics_keys: list[str] = []
    diagnostics: dict[str, Any] = {}
    metrics: dict[str, Any] = {"relaxation": relaxation}

    rules = _validation_rules(runtime)
    _record_check(
        checks,
        physics_keys,
        "validation.contract_present",
        bool(rules),
    )
    _record_check(
        checks,
        physics_keys,
        "policy.pure_simulation_after_t0",
        bool(runtime.plan["policy"]["initial_state_only_from_video"])
        and not runtime.plan["policy"]["source_video_access_during_rollout"]
        and not runtime.plan["policy"]["future_pose_replay"],
    )

    first = records[0]
    for name, entity in runtime.manifest["resolved_entities"].items():
        if entity["physics_type"] != "rigid":
            continue
        state = first["rigid"][name]
        position_error = float(
            np.linalg.norm(
                np.asarray(state["position_m"]) - np.asarray(entity["position_m"])
            )
        )
        quaternion_error = _quat_error_rad(
            np.asarray(state["quaternion_wxyz"]),
            np.asarray(entity["quaternion_wxyz"]),
        )
        linear_velocity_error = float(
            np.linalg.norm(
                np.asarray(state["linear_velocity_m_s"])
                - np.asarray(entity["initial_state"]["linear_velocity_m_s"])
            )
        )
        angular_velocity_error = float(
            np.linalg.norm(
                np.asarray(state["angular_velocity_rad_s"])
                - np.asarray(entity["initial_state"]["angular_velocity_rad_s"])
            )
        )
        metrics.setdefault("initial_state_error", {})[name] = {
            "position_m": position_error,
            "rotation_rad": quaternion_error,
            "linear_velocity_m_s": linear_velocity_error,
            "angular_velocity_rad_s": angular_velocity_error,
        }
        _record_check(
            checks,
            physics_keys,
            f"{name}.initial_position",
            position_error <= 1e-5,
        )
        _record_check(
            checks,
            physics_keys,
            f"{name}.initial_orientation",
            quaternion_error <= 1e-5,
        )
        _record_check(
            checks,
            physics_keys,
            f"{name}.initial_linear_velocity",
            linear_velocity_error <= 1e-5,
        )
        _record_check(
            checks,
            physics_keys,
            f"{name}.initial_angular_velocity",
            angular_velocity_error <= 1e-5,
        )

    for name, item in relaxation["deformables"].items():
        _record_check(
            checks,
            physics_keys,
            f"{name}.initial_surface_alignment",
            item["surface_p95_displacement_m"] <= 0.005,
        )
        _record_check(
            checks,
            physics_keys,
            f"{name}.t0_velocity_reset",
            bool(item["velocity_reset_at_t0"]),
        )

    all_numeric: list[float] = []
    for record in records:
        for state in record["rigid"].values():
            all_numeric.extend(state["position_m"])
            all_numeric.extend(state["quaternion_wxyz"])
            all_numeric.extend(state["linear_velocity_m_s"])
            all_numeric.extend(state["angular_velocity_rad_s"])
        for state in record["deformable"].values():
            all_numeric.extend(state["bounds_m"][0])
            all_numeric.extend(state["bounds_m"][1])
            all_numeric.append(state["max_surface_speed_m_s"])
    numeric = np.asarray(all_numeric, dtype=float)
    _record_check(
        checks, physics_keys, "states.finite", bool(np.isfinite(numeric).all())
    )
    _record_check(
        checks,
        physics_keys,
        "states.bounded",
        bool(np.abs(numeric).max(initial=0.0) < 50.0),
    )

    for name in runtime.deformable:
        minimum_z = min(
            record["deformable"][name]["bounds_m"][0][2] for record in records
        )
        inverted = max(
            record["deformable"][name]["inverted_tetrahedra"] for record in records
        )
        minimum_volume = min(
            record["deformable"][name]["minimum_oriented_tet_volume_m3"]
            for record in records
        )
        inversion_fraction = float(
            inverted / max(1, runtime.deformable[name].n_elements)
        )
        metrics.setdefault("deformable_quality", {})[name] = {
            "minimum_surface_z_m": float(minimum_z),
            "maximum_inverted_tetrahedra": int(inverted),
            "maximum_inverted_fraction": inversion_fraction,
            "minimum_oriented_volume_m3": float(minimum_volume),
            "acceptance_role": "diagnostic_only_for_preview; training_blocker",
        }
        diagnostics[f"{name}.local_tet_inversion"] = {
            "observed": bool(inverted > 0),
            "maximum_inverted_fraction": inversion_fraction,
            "accepted_for_current_preview": inversion_fraction <= 0.01,
        }
        support_name = runtime.manifest["resolved_entities"][name]["deformable"][
            "boundary_conditions"
        ]["support_entity"]
        support_top = world_bounds(runtime.manifest["resolved_entities"][support_name])[
            1, 2
        ]
        _record_check(
            checks,
            physics_keys,
            f"{name}.support_penetration",
            minimum_z >= support_top - 0.006,
        )

    for rule in rules:
        if rule["type"] == "soft_impact":
            _validate_soft_impact(
                runtime, config, records, rule, checks, physics_keys, metrics
            )
        elif rule["type"] == "rolling_transition":
            _validate_rolling_transition(
                runtime, records, rule, checks, physics_keys, metrics
            )
        else:
            raise ValueError(f"Unsupported physics validation type: {rule['type']!r}")

    if config.render:
        checks["render.preview_exists"] = (config.sim_dir / "preview.png").is_file()
        checks["render.video_exists"] = (config.sim_dir / "simulation.mp4").is_file()
        checks["render.video_probe"] = bool(
            video_probe and int(video_probe.get("nb_frames", 0)) >= 2
        )

    physics_validated = bool(physics_keys) and all(checks[key] for key in physics_keys)
    training_blockers = runtime.manifest.get("physics_validation", {}).get(
        "training_blockers",
        [
            "Material parameters are monocular engineering priors, not identified measurements.",
            "Only the nominal scene has been simulated; parameter-sweep convergence is not established.",
        ],
    )
    return {
        "schema_version": "video2scene.genesis-validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "physics_validated": physics_validated,
        "training_ready": False,
        "training_blockers": training_blockers,
    }


def _write_trace_npz(output: Path, records: list[dict[str, Any]]) -> None:
    arrays: dict[str, np.ndarray] = {
        "time_s": np.asarray([record["time_s"] for record in records], dtype=np.float64)
    }
    for name in records[0]["rigid"]:
        arrays[f"{name}_position_m"] = np.asarray(
            [record["rigid"][name]["position_m"] for record in records]
        )
        arrays[f"{name}_quaternion_wxyz"] = np.asarray(
            [record["rigid"][name]["quaternion_wxyz"] for record in records]
        )
        arrays[f"{name}_linear_velocity_m_s"] = np.asarray(
            [record["rigid"][name]["linear_velocity_m_s"] for record in records]
        )
        arrays[f"{name}_angular_velocity_rad_s"] = np.asarray(
            [record["rigid"][name]["angular_velocity_rad_s"] for record in records]
        )
        arrays[f"{name}_bottom_z_m"] = np.asarray(
            [record["rigid"][name]["bottom_z_m"] for record in records]
        )
    for relation_id, first_relation in records[0]["relations"].items():
        for field, value in first_relation.items():
            if not isinstance(value, (int, float)):
                continue
            arrays[f"relation/{relation_id}/{field}"] = np.asarray(
                [record["relations"][relation_id][field] for record in records]
            )
    np.savez_compressed(output / "trace.npz", **arrays)


def _run_prepared(config: SimulationRunConfig, frames_dir: Path) -> dict[str, Any]:
    runtime: RuntimeScene | None = None
    started = time.perf_counter()
    try:
        runtime = build_runtime(config)
        write_json(config.sim_dir / "migration_plan.json", runtime.plan)
        relaxation = _relax_and_initialize(runtime, config)
        patches = _make_patch_indices(runtime)
        records = [_sample_state(runtime, 0.0, patches)]
        frame_paths: list[Path] = []
        frame_times: list[float] = []
        if config.render:
            frame = frames_dir / "frame_0000.png"
            render_frame(runtime, frame)
            frame_paths.append(frame)
            frame_times.append(0.0)
            Image.open(frame).save(config.sim_dir / "preview.png")

        n_steps = int(math.ceil(config.duration_s / config.dt))
        render_every = config.simulation_hz // config.fps
        for step in range(1, n_steps + 1):
            runtime.scene.step(update_visualizer=False)
            sync_visual_followers(runtime)
            if step % render_every == 0 or step == n_steps:
                time_s = step * config.dt
                records.append(_sample_state(runtime, time_s, patches))
                if config.render:
                    frame = frames_dir / f"frame_{len(frame_paths):04d}.png"
                    render_frame(runtime, frame)
                    frame_paths.append(frame)
                    frame_times.append(time_s)

        write_json(
            config.sim_dir / "trace.json",
            {"schema_version": "video2scene.trace.v1", "records": records},
        )
        _write_trace_npz(config.sim_dir, records)
        video_probe = None
        if config.render:
            video_probe = encode_video(
                frames_dir, config.sim_dir / "simulation.mp4", config.fps
            )
            write_comparison(config)
            write_contact_sheet(config, frame_paths, frame_times)
            write_source_simulation_grid(config, frame_paths, frame_times)
        validation = _validate_run(runtime, config, relaxation, records, video_probe)
        write_json(config.sim_dir / "validation.json", validation)
        if config.render:
            _write_render_review(config, validation)
        run_manifest = {
            "schema_version": "video2scene.genesis-run.v1",
            "scene_id": runtime.manifest["scene_id"],
            "phase": 2,
            "genesis_version": getattr(runtime.gs, "__version__", "unknown"),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "timing": {
                "build_seconds": runtime.build_seconds,
                "total_seconds": time.perf_counter() - started,
                "simulated_seconds": n_steps * config.dt,
                "relaxation_seconds": relaxation["duration_s"],
            },
            "discretization": {
                name: {
                    "vertices": int(entity.n_vertices),
                    "tetrahedra": int(entity.n_elements),
                    "render_vertices": int(entity.n_vverts),
                    "render_faces": int(entity.n_vfaces),
                }
                for name, entity in runtime.deformable.items()
            },
            "artifacts": {
                name: {
                    "path": name,
                    "sha256": sha256_file(config.sim_dir / name),
                }
                for name in (
                    [
                        "migration_plan.json",
                        "trace.json",
                        "trace.npz",
                        "validation.json",
                    ]
                    + (
                        [
                            "preview.png",
                            "comparison.jpg",
                            "contact_sheet.jpg",
                            "source_simulation_grid.jpg",
                            "simulation.mp4",
                            "render_review.json",
                        ]
                        if config.render
                        else []
                    )
                )
                if (config.sim_dir / name).is_file()
            },
            "video_probe": video_probe,
            "physics_validated": validation["physics_validated"],
            "training_ready": False,
        }
        write_json(config.sim_dir / "run_manifest.json", run_manifest)
        return validation
    finally:
        if runtime is not None:
            runtime.gs.destroy()


def run(config: SimulationRunConfig) -> dict[str, Any]:
    """Execute one resolved simulation and persist a durable run log."""
    validate_simulation_options(config)
    validate_simulation_inputs(config)
    frames_dir = prepare_simulation_output(config.sim_dir, config.overwrite)
    with stage_log(config.sim_dir / "run.log"):
        logger.info(
            "[Phase 2] Start: input={} output={} environments={}",
            config.blender_dir,
            config.sim_dir,
            config.num_envs,
        )
        validation = _run_prepared(config, frames_dir)
        logger.info(
            "[Phase 2] Complete: passed={} physics_validated={}",
            validation["passed"],
            validation["physics_validated"],
        )
        return validation


def main() -> None:
    config = resolve_simulation_command(tyro.cli(SimulationCommandConfig))
    try:
        validation = run(config)
    except Exception:
        logger.exception("[Phase 2] Simulation failed")
        raise
    failures = [name for name, passed in validation["checks"].items() if not passed]
    logger.info(
        "{}",
        json.dumps(
            {
                "output": str(config.sim_dir),
                "passed": validation["passed"],
                "physics_validated": validation["physics_validated"],
                "failed_checks": failures,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
