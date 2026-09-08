#!/usr/bin/env python
"""Audit code/skill drift and evidence that should drive safe self-evolution."""

from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger
import tyro

from inspect_repo import Args as InspectArgs
from inspect_repo import inspect as inspect_repository
from video2scene import __version__
from video2scene.layout import DEFAULT_SCENE_ID, PROJECT_ROOT
from video2scene.spec import DEFAULT_SPEC, read_json

BASELINE_SCHEMA = "video2scene.evolution-baseline.v1"
DEFAULT_BASELINE = PROJECT_ROOT / ".agent/video2scene/evolution-baseline.json"
TRACKED_MODULES = (
    "video2scene/layout.py",
    "video2scene/spec.py",
    "video2scene/build.py",
    "video2scene/assets.py",
    "video2scene/validate.py",
    "video2scene/genesis_plan.py",
    "video2scene/simulation_config.py",
    "video2scene/simulation_runtime.py",
    "video2scene/simulate.py",
    "video2scene/domain_randomization.py",
    "video2scene/dataset_simulation.py",
    "video2scene/dataset_artifacts.py",
    "video2scene/generate_dataset.py",
    ".agent/video2scene/scripts/inspect_repo.py",
    ".agent/video2scene/scripts/evolution_audit.py",
)
SCENE_SENSITIVE_MODULES = {
    "phase1": ("video2scene/build.py", "video2scene/spec.py"),
    "phase2": (
        "video2scene/build.py",
        "video2scene/spec.py",
        "video2scene/simulate.py",
    ),
    "phase3": (
        "video2scene/build.py",
        "video2scene/spec.py",
        "video2scene/simulate.py",
        "video2scene/dataset_simulation.py",
    ),
}
REFERENCE_ENTITY_IDS = {"ball", "block", "cushion_left", "cushion_right"}
SCENE_TOKENS = (
    "ball_contacts_left_cushion",
    "block_contacts_right_cushion",
    "retained_on_target",
    "block.tipped",
)
REQUIRED_SKILL_FILES = (
    "SKILL.md",
    "agents/openai.yaml",
    "references/phase-playbook.md",
    "references/repository-contract.md",
    "references/self-evolution.md",
    "references/validation.md",
    "scripts/evolution_audit.py",
    "scripts/inspect_repo.py",
)


@dataclass(frozen=True)
class Args:
    """Audit a scene task against the live code and reviewed skill baseline."""

    spec: Path = DEFAULT_SPEC
    output_root: Path | None = None
    target_phase: Literal["auto", "phase1", "phase2", "phase3"] = "auto"
    baseline: Path = DEFAULT_BASELINE
    report: Path | None = None
    fail_on_findings: bool = False
    refresh_baseline: bool = False
    reviewed: bool = False


def expression(node: ast.expr | None) -> str | None:
    """Render one AST expression in a stable, readable form."""
    return ast.unparse(node) if node is not None else None


def argument_contract(argument: ast.arg, default: ast.expr | None) -> dict[str, Any]:
    """Describe one argument, including annotations and required/default status."""
    return {
        "name": argument.arg,
        "annotation": expression(argument.annotation),
        "default": expression(default) if default is not None else "<required>",
    }


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Return a stable structural signature without coupling to implementation text."""
    positional = [*node.args.posonlyargs, *node.args.args]
    positional_defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(node.args.defaults)
    ) + list(node.args.defaults)
    return {
        "positional": [
            argument_contract(argument, default)
            for argument, default in zip(positional, positional_defaults, strict=True)
        ],
        "keyword_only": [
            argument_contract(argument, default)
            for argument, default in zip(
                node.args.kwonlyargs, node.args.kw_defaults, strict=True
            )
        ],
        "vararg": argument_contract(node.args.vararg, None)
        if node.args.vararg
        else None,
        "kwarg": argument_contract(node.args.kwarg, None) if node.args.kwarg else None,
        "returns": expression(node.returns),
    }


def module_contract(path: Path) -> dict[str, Any]:
    """Extract public top-level functions, classes and annotated class fields."""
    tree = ast.parse(path.read_text(), filename=str(path))
    functions: dict[str, Any] = {}
    classes: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not node.name.startswith("_"):
            functions[node.name] = function_signature(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            fields = {
                item.target.id: {
                    "annotation": expression(item.annotation),
                    "default": expression(item.value),
                }
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
            methods = {
                item.name: function_signature(item)
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not item.name.startswith("_")
            }
            classes[node.name] = {"fields": fields, "methods": methods}
    return {"functions": functions, "classes": classes}


def skill_structure(skill_root: Path) -> dict[str, Any]:
    """Capture frontmatter identity and Markdown section structure."""
    result: dict[str, Any] = {}
    for path in sorted(skill_root.rglob("*.md")):
        relative = str(path.relative_to(skill_root))
        lines = path.read_text().splitlines()
        result[relative] = {
            "headings": [line.strip() for line in lines if line.startswith("#")]
        }
        if relative == "SKILL.md":
            frontmatter = {}
            if lines and lines[0] == "---":
                for line in lines[1:]:
                    if line == "---":
                        break
                    key, separator, value = line.partition(":")
                    if separator:
                        frontmatter[key.strip()] = value.strip()
            result[relative]["frontmatter"] = frontmatter
    return result


def collect_contract() -> dict[str, Any]:
    """Collect the stable repository surface the skill depends on."""
    schema_path = PROJECT_ROOT / "schemas/scene.schema.json"
    schema = read_json(schema_path)
    entity_properties = schema["properties"]["entities"]["additionalProperties"][
        "properties"
    ]
    geometry_enum = sorted(
        entity_properties["geometry"]["properties"]["generator"]["enum"]
    )
    physics_enum = sorted(entity_properties["physics"]["properties"]["type"]["enum"])
    skill_root = PROJECT_ROOT / ".agent/video2scene"
    return {
        "schema_version": BASELINE_SCHEMA,
        "package_version": __version__,
        "scene_schema": {
            "id": schema.get("$id"),
            "scene_version": schema["properties"]["schema_version"]["const"],
            "top_level_required": sorted(schema["required"]),
            "geometry_generators": geometry_enum,
            "physics_types": physics_enum,
        },
        "modules": {
            relative: module_contract(PROJECT_ROOT / relative)
            for relative in TRACKED_MODULES
        },
        "skill_resources": {
            relative: (skill_root / relative).is_file()
            for relative in REQUIRED_SKILL_FILES
        },
        "skill_structure": skill_structure(skill_root),
    }


def baseline_diff(current: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    """Compare the current structural contract with the reviewed baseline."""
    if not baseline_path.is_file():
        return {
            "exists": False,
            "matches": False,
            "diff": [f"Missing baseline: {baseline_path}"],
        }
    baseline = read_json(baseline_path)
    current_text = json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True)
    baseline_text = json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True)
    diff = list(
        difflib.unified_diff(
            baseline_text.splitlines(),
            current_text.splitlines(),
            fromfile=str(baseline_path),
            tofile="live-contract",
            lineterm="",
        )
    )
    return {"exists": True, "matches": not diff, "diff": diff[:400]}


def markdown_link_findings(skill_root: Path) -> list[dict[str, Any]]:
    """Find broken relative Markdown links inside the skill."""
    findings = []
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for path in sorted(skill_root.rglob("*.md")):
        for target in pattern.findall(path.read_text()):
            target_path = target.split("#", 1)[0]
            if not target_path or "://" in target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                findings.append(
                    {
                        "file": str(path.relative_to(PROJECT_ROOT)),
                        "target": target,
                    }
                )
    return findings


def unfinished_markers() -> list[dict[str, Any]]:
    """Report explicit unfinished markers in tracked code and skill resources."""
    findings = []
    paths = [PROJECT_ROOT / relative for relative in TRACKED_MODULES]
    paths.extend((PROJECT_ROOT / ".agent/video2scene").rglob("*.md"))
    for path in sorted(set(paths)):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\b(?:TODO|FIXME|XXX)\s*:", line):
                findings.append(
                    {
                        "file": str(path.relative_to(PROJECT_ROOT)),
                        "line": line_number,
                        "text": line.strip(),
                    }
                )
    return findings


def scene_hardcoding(modules: tuple[str, ...]) -> list[dict[str, Any]]:
    """Locate reference-scene string constants in modules expected to evolve."""
    findings = []
    for relative in modules:
        path = PROJECT_ROOT / relative
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            matched = sorted(token for token in SCENE_TOKENS if token in node.value)
            if node.value in REFERENCE_ENTITY_IDS:
                matched.append(node.value)
            if matched:
                findings.append(
                    {
                        "file": relative,
                        "line": getattr(node, "lineno", None),
                        "tokens": sorted(set(matched)),
                        "value": node.value[:180],
                    }
                )
    return findings[:100]


def resolve_target_phase(requested: str, coverage: dict[str, Any]) -> str:
    """Resolve the audit phase from an explicit request or existing artifacts."""
    if requested != "auto":
        return requested
    if coverage.get("dataset"):
        return "phase3"
    if any(coverage["simulation"].values()):
        return "phase2"
    if any(coverage["blender"].values()):
        return "phase2"
    return "phase1"


def recent_runtime_evidence(paths: list[Path]) -> list[dict[str, Any]]:
    """Collect error-like lines near the end of current durable logs."""
    evidence = []
    pattern = re.compile(
        r"(?:\bERROR\b|Traceback|Exception|MigrationError|failed)", re.I
    )
    for path in paths:
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 256 * 1024))
            lines = stream.read().decode(errors="replace").splitlines()[-400:]
        matches = [line.strip() for line in lines if pattern.search(line)]
        if matches:
            evidence.append(
                {
                    "path": str(path),
                    "matches": matches[-20:],
                    "interpretation": "Triage against current artifacts; log text may describe an already repaired attempt.",
                }
            )
    return evidence


def recommendations(
    *,
    scene_id: str,
    drift: dict[str, Any],
    repo_invalid: bool,
    broken_links: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    hardcoding: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> list[str]:
    """Turn detected evidence into bounded next actions."""
    result = []
    if not drift["matches"]:
        result.append(
            "Review the live API/schema drift, update code-linked skill guidance, run tests and a real path, then refresh the baseline with --reviewed."
        )
    if repo_invalid:
        result.append(
            "Reproduce each failed validation, migration error or corrupt success marker before changing thresholds or artifacts."
        )
    if broken_links:
        result.append("Repair broken skill references and rerun the skill validator.")
    if markers:
        result.append(
            "Resolve or intentionally document unfinished TODO/FIXME/XXX markers before claiming the affected path complete."
        )
    if scene_id != DEFAULT_SCENE_ID and hardcoding:
        result.append(
            "Generalize or isolate reference-scene entity/event hardcoding before applying nominal and batch validation to this scene."
        )
    if not any(coverage["blender"].values()):
        result.append(
            "No Phase-1 artifacts are present; do not claim visual handoff validation."
        )
    if not any(coverage["simulation"].values()):
        result.append(
            "No Phase-2 artifacts are present; do not claim physics validation."
        )
    return result


def audit(args: Args) -> tuple[dict[str, Any], bool]:
    """Run structural, skill, artifact and recent-log audits."""
    spec = read_json(args.spec.resolve())
    repository, repo_invalid = inspect_repository(
        InspectArgs(spec=args.spec, output_root=args.output_root)
    )
    coverage = {
        **repository["artifacts"],
        "dataset": repository["dataset"]["batch_count"] > 0,
    }
    target_phase = resolve_target_phase(args.target_phase, coverage)
    current_contract = collect_contract()
    drift = baseline_diff(current_contract, args.baseline.resolve())
    skill_root = PROJECT_ROOT / ".agent/video2scene"
    broken_links = markdown_link_findings(skill_root)
    markers = unfinished_markers()
    hardcoding = scene_hardcoding(SCENE_SENSITIVE_MODULES[target_phase])
    layout = repository["layout"]
    log_paths = [
        Path(layout["sim"]) / "run.log",
        Path(layout["batches"]) / "generate.log",
    ]
    batch_logs = sorted(Path(layout["batches"]).glob("batch_*/sim/run.log"))[-5:]
    runtime_evidence = recent_runtime_evidence([*log_paths, *batch_logs])
    missing_resources = sorted(
        name
        for name, present in current_contract["skill_resources"].items()
        if not present
    )
    hardcoding_blocks_scene = spec["scene_id"] != DEFAULT_SCENE_ID and bool(hardcoding)
    blocking = {
        "structural_drift": not drift["matches"],
        "repository_invalid": repo_invalid,
        "broken_skill_links": broken_links,
        "missing_skill_resources": missing_resources,
        "scene_specific_hardcoding": hardcoding_blocks_scene,
    }
    has_blocking = any(
        value if isinstance(value, bool) else bool(value) for value in blocking.values()
    )
    report = {
        "schema_version": "video2scene.evolution-report.v1",
        "scene_id": spec["scene_id"],
        "target_phase": target_phase,
        "baseline": {
            "path": str(args.baseline.resolve()),
            **drift,
        },
        "blocking": blocking,
        "runtime_evidence": runtime_evidence,
        "technical_debt": {
            "unfinished_markers": markers,
            "reference_scene_hardcoding": hardcoding,
            "hardcoding_blocks_this_scene": hardcoding_blocks_scene,
        },
        "coverage": repository["artifacts"],
        "repository_snapshot": repository,
        "recommendations": recommendations(
            scene_id=spec["scene_id"],
            drift=drift,
            repo_invalid=repo_invalid,
            broken_links=broken_links,
            markers=markers,
            hardcoding=hardcoding,
            coverage=repository["artifacts"],
        ),
    }
    return report, has_blocking


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def main(args: Args) -> int:
    """Run the audit, optionally update its reviewed structural baseline."""
    baseline_path = args.baseline.resolve()
    if args.refresh_baseline:
        if not args.reviewed:
            logger.error("--refresh-baseline requires --reviewed")
            return 3
        write_json(baseline_path, collect_contract())
        logger.info("Refreshed reviewed structural baseline at {}", baseline_path)

    try:
        report, has_blocking = audit(args)
    except Exception:
        logger.exception("Self-evolution audit failed")
        return 1

    if args.report is not None:
        write_json(args.report.resolve(), report)
        logger.info("Wrote self-evolution report to {}", args.report.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if args.fail_on_findings and has_blocking:
        logger.error("Blocking self-evolution findings remain")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
