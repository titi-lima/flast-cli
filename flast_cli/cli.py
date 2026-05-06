"""Command-line interface for the FLAST wrapper."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

from .adapters import PytestAdapter
from .constants import DEFAULT_MIN_OBSERVATIONS, DEFAULT_RUNS, SCHEMA_VERSION
from .corpus import iter_manifest_sources, rebuild_corpus
from .diagnostics import build_diagnostics
from .features import build_feature_text
from .manifests import load_labels, load_manifest, load_report, write_report
from .model import TrainingParams, load_trained_artifacts, predict_scores, train_model
from .reporting import render_report, write_optional_output
from .schemas import PredictionRecord, PredictionReport
from .workspace import (
    ensure_workspace,
    latest_report_path,
    list_workspace_manifests,
    load_config,
    report_path,
    run_manifest_path,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flast",
        description="Headless CLI wrapper for FLAST flaky-test detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Initialize a FLAST workspace")
    init_sub = init_parser.add_subparsers(dest="framework", required=True)
    init_pytest = init_sub.add_parser("pytest", help="Initialize Pytest support")
    init_pytest.add_argument("--project-dir", default=".", help="Path to project root")

    observe_parser = sub.add_parser("observe", help="Observe repeated test runs")
    observe_sub = observe_parser.add_subparsers(dest="framework", required=True)
    observe_pytest = observe_sub.add_parser("pytest", help="Observe Pytest runs")
    observe_pytest.add_argument("--project-dir", default=".", help="Path to project root")
    observe_pytest.add_argument(
        "--runs",
        type=int,
        default=None,
        help=f"Number of repeated Pytest invocations (default: {DEFAULT_RUNS})",
    )
    observe_pytest.add_argument(
        "--min-observations",
        type=int,
        default=None,
        help=f"Minimum observations required for non_flaky labels (default: {DEFAULT_MIN_OBSERVATIONS})",
    )
    observe_pytest.add_argument(
        "--reuse-history",
        action="append",
        default=[],
        help="Manifest file or directory to import into the managed corpus",
    )
    observe_pytest.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument to forward to pytest (repeatable)",
    )

    train_parser = sub.add_parser("train", help="Train a FLAST model from the corpus")
    train_parser.add_argument("--project-dir", default=".", help="Path to project root")
    train_parser.add_argument("--k", type=int, default=7, help="Number of neighbors")
    train_parser.add_argument("--sigma", type=float, default=0.5, help="Threshold sigma")
    train_parser.add_argument(
        "--eps",
        type=float,
        default=0.3,
        help="Projection epsilon; use 0 to disable projection",
    )
    train_parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric",
    )
    train_parser.add_argument(
        "--weights",
        default="distance",
        choices=["distance", "uniform"],
        help="Neighbor weighting strategy",
    )
    train_parser.add_argument(
        "--algorithm",
        default="brute",
        choices=["brute", "auto"],
        help="Underlying kNN algorithm",
    )

    predict_parser = sub.add_parser("predict", help="Collect and score the current suite")
    predict_sub = predict_parser.add_subparsers(dest="framework", required=True)
    predict_pytest = predict_sub.add_parser("pytest", help="Predict Pytest flakiness")
    predict_pytest.add_argument("--project-dir", default=".", help="Path to project root")
    predict_pytest.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    predict_pytest.add_argument("--output", help="Optional output path for rendered report")
    predict_pytest.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument to forward to pytest (repeatable)",
    )

    diagnostics_parser = sub.add_parser(
        "diagnostics",
        help="Emit editor-oriented diagnostics for the current suite",
    )
    diagnostics_sub = diagnostics_parser.add_subparsers(dest="framework", required=True)
    diagnostics_pytest = diagnostics_sub.add_parser(
        "pytest",
        help="Emit Pytest diagnostics",
    )
    diagnostics_pytest.add_argument("--project-dir", default=".", help="Path to project root")
    diagnostics_pytest.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format",
    )
    diagnostics_pytest.add_argument("--output", help="Optional output path for rendered diagnostics")
    diagnostics_pytest.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument to forward to pytest (repeatable)",
    )

    report_parser = sub.add_parser("report", help="Render an existing report")
    report_parser.add_argument("--project-dir", default=".", help="Path to project root")
    report_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    group = report_parser.add_mutually_exclusive_group()
    group.add_argument("--latest", action="store_true", help="Render the latest report")
    group.add_argument("--run-id", help="Render the report for a specific run id")

    return parser


def _project_root(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _adapter_for(framework: str) -> PytestAdapter:
    if framework != "pytest":
        raise ValueError(f"Unsupported framework: {framework}")
    return PytestAdapter()


def _default_from_config(project_root: Path, section: str, key: str, fallback):
    config = load_config(project_root)
    return config.get(section, {}).get(key, fallback)


def _load_workspace_manifests(project_root: Path) -> List:
    return [load_manifest(path) for path in list_workspace_manifests(project_root)]


def _run_pytest_observation(
    *,
    adapter: PytestAdapter,
    project_root: Path,
    pytest_args: Sequence[str],
) -> Path:
    run_id = str(uuid.uuid4())
    manifest_path = run_manifest_path(project_root, run_id)
    command = adapter.build_observe_invocation(
        project_root=project_root,
        manifest_path=manifest_path,
        pytest_args=pytest_args,
    )
    returncode, stdout, stderr = adapter.run(project_root=project_root, command=command)
    if returncode not in (0, 1, 5):
        raise RuntimeError(
            f"pytest observation failed with exit code {returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    if not manifest_path.exists():
        raise RuntimeError("Pytest observation finished without producing a manifest.")
    return manifest_path


def _run_pytest_collection(
    *,
    adapter: PytestAdapter,
    project_root: Path,
    pytest_args: Sequence[str],
) -> Path:
    run_id = str(uuid.uuid4())
    manifest_path = run_manifest_path(project_root, run_id)
    command = adapter.build_collect_invocation(
        project_root=project_root,
        manifest_path=manifest_path,
        pytest_args=pytest_args,
    )
    returncode, stdout, stderr = adapter.run(project_root=project_root, command=command)
    if returncode not in (0, 5):
        raise RuntimeError(
            f"pytest collection failed with exit code {returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    if not manifest_path.exists():
        raise RuntimeError("Pytest collection finished without producing a manifest.")
    return manifest_path


def run_init(args: argparse.Namespace) -> int:
    project_root = _project_root(args.project_dir)
    paths = ensure_workspace(project_root)
    print(f"Initialized FLAST workspace at {paths.workspace_dir}")
    print("Pytest integration is wrapper-driven; no pytest.ini changes were made.")
    return 0


def run_observe(args: argparse.Namespace) -> int:
    project_root = _project_root(args.project_dir)
    ensure_workspace(project_root)
    adapter = _adapter_for(args.framework)

    imported_manifests = []
    for manifest_source in iter_manifest_sources(args.reuse_history):
        manifest = adapter.load_manifest(manifest_source)
        destination = run_manifest_path(project_root, manifest.run_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_text(manifest_source.read_text(encoding="utf-8"), encoding="utf-8")
        imported_manifests.append(destination)

    run_count = (
        args.runs
        if args.runs is not None
        else _default_from_config(project_root, "observe", "default_runs", DEFAULT_RUNS)
    )
    for _ in range(run_count):
        _run_pytest_observation(
            adapter=adapter,
            project_root=project_root,
            pytest_args=args.pytest_arg,
        )

    min_observations = (
        args.min_observations
        if args.min_observations is not None
        else _default_from_config(
            project_root,
            "observe",
            "min_observations",
            DEFAULT_MIN_OBSERVATIONS,
        )
    )
    manifests = _load_workspace_manifests(project_root)
    observations, labels = rebuild_corpus(
        project_root=project_root,
        manifests=manifests,
        min_observations=min_observations,
    )
    counts = {"flaky": 0, "non_flaky": 0, "unlabeled": 0}
    for label in labels:
        counts[label.label] = counts.get(label.label, 0) + 1
    print(
        "Observed {runs} run(s); corpus now tracks {tests} test(s) "
        "({flaky} flaky, {non_flaky} non_flaky, {unlabeled} unlabeled).".format(
            runs=run_count,
            tests=len(observations),
            flaky=counts.get("flaky", 0),
            non_flaky=counts.get("non_flaky", 0),
            unlabeled=counts.get("unlabeled", 0),
        )
    )
    return 0


def run_train(args: argparse.Namespace) -> int:
    project_root = _project_root(args.project_dir)
    ensure_workspace(project_root)
    labels = load_labels(ensure_workspace(project_root).labels_path)
    if not labels:
        raise FileNotFoundError(
            f"No labels found under {ensure_workspace(project_root).labels_path}; run `flast observe pytest` first."
        )

    adapter = _adapter_for("pytest")
    feature_texts = []
    for label in labels:
        feature = adapter.resolve_feature_input(label.test_id, Path(label.project_root))
        feature_texts.append(build_feature_text(feature))

    metadata = train_model(
        project_root=project_root,
        labels=labels,
        feature_texts=feature_texts,
        params=TrainingParams(
            k=args.k,
            sigma=args.sigma,
            eps=args.eps,
            metric=args.metric,
            weights=args.weights,
            algorithm=args.algorithm,
        ),
    )
    print(
        "Trained FLAST model with {size} labeled tests "
        "({flaky} flaky, {non_flaky} non_flaky).".format(
            size=metadata["corpus_size"],
            flaky=metadata["label_counts"]["flaky"],
            non_flaky=metadata["label_counts"]["non_flaky"],
        )
    )
    print(
        "Parameters: k={k}, sigma={sigma}, eps={eps}, metric={metric}, weights={weights}, algorithm={algorithm}".format(
            **metadata["params"]
        )
    )
    return 0


def run_predict(args: argparse.Namespace) -> int:
    report = create_prediction_report(
        project_root=_project_root(args.project_dir),
        framework=args.framework,
        pytest_args=args.pytest_arg,
    )
    rendered = render_report(report, output_format=args.format)
    print(rendered)
    if args.output:
        write_optional_output(Path(args.output).expanduser().resolve(), rendered)
    return 0


def create_prediction_report(
    *,
    project_root: Path,
    framework: str,
    pytest_args: Sequence[str],
) -> PredictionReport:
    ensure_workspace(project_root)
    adapter = _adapter_for(framework)
    manifest_path = _run_pytest_collection(
        adapter=adapter,
        project_root=project_root,
        pytest_args=pytest_args,
    )
    manifest = adapter.load_manifest(manifest_path)
    artifacts = load_trained_artifacts(project_root)
    feature_inputs = [
        adapter.resolve_feature_input(item.test_id, Path(manifest.project_root))
        for item in manifest.collected
    ]
    feature_texts = [build_feature_text(item) for item in feature_inputs]
    scores, predictions = predict_scores(artifacts, feature_texts)
    records = [
        PredictionRecord(
            test_id=feature.test_id,
            score=float(score),
            predicted_label="flaky" if prediction == 1 else "non_flaky",
            class_name=feature.class_name,
            test_name=feature.test_name,
            file_path=str(feature.file_path) if feature.file_path else None,
        )
        for feature, score, prediction in zip(feature_inputs, scores, predictions)
    ]
    report = PredictionReport(
        schema_version=SCHEMA_VERSION,
        run_id=manifest.run_id,
        framework=manifest.framework,
        project_root=manifest.project_root,
        model_metadata=artifacts.metadata,
        predictions=records,
    )
    write_report(report_path(project_root, manifest.run_id), report)
    return report


def run_diagnostics(args: argparse.Namespace) -> int:
    report = create_prediction_report(
        project_root=_project_root(args.project_dir),
        framework=args.framework,
        pytest_args=args.pytest_arg,
    )
    rendered = json.dumps(build_diagnostics(report), indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        write_optional_output(Path(args.output).expanduser().resolve(), rendered)
    return 0


def run_report(args: argparse.Namespace) -> int:
    project_root = _project_root(args.project_dir)
    ensure_workspace(project_root)
    target = (
        report_path(project_root, args.run_id)
        if args.run_id
        else latest_report_path(project_root)
    )
    report = load_report(target)
    print(render_report(report, output_format=args.format))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return run_init(args)
        if args.command == "observe":
            return run_observe(args)
        if args.command == "train":
            return run_train(args)
        if args.command == "predict":
            return run_predict(args)
        if args.command == "diagnostics":
            return run_diagnostics(args)
        if args.command == "report":
            return run_report(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
