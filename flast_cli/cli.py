"""Command-line interface for the FLAST wrapper."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

from .adapters import FrameworkAdapter, get_adapter, supported_frameworks
from .constants import DEFAULT_MIN_OBSERVATIONS, DEFAULT_RUNS, SCHEMA_VERSION
from .corpus import iter_manifest_sources, rebuild_corpus
from .diagnostics import build_diagnostics
from .features import build_feature_text, detect_static_signals
from .manifests import load_labels, load_manifest, load_report, write_report
from .model import ExplainedPrediction, TrainingParams, load_trained_artifacts, predict_with_explanations, train_model
from .reporting import render_report, write_optional_output
from .schemas import FeatureInput, LabelRecord, PredictionRecord, PredictionReport
from .workspace import (
    ensure_workspace,
    latest_report_path,
    list_workspace_manifests,
    load_config,
    report_path,
    run_manifest_path,
)


def _add_framework_args(parser: argparse.ArgumentParser, framework: str) -> None:
    flags = ["--framework-arg"]
    if framework == "pytest":
        flags.append("--pytest-arg")
    parser.add_argument(
        *flags,
        dest="framework_args",
        action="append",
        default=[],
        help="Additional argument to forward to the test framework (repeatable)",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flast",
        description="Headless CLI wrapper for FLAST flaky-test detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Initialize a FLAST workspace")
    init_sub = init_parser.add_subparsers(dest="framework", required=True)
    for framework in supported_frameworks():
        init_framework = init_sub.add_parser(framework, help=f"Initialize {framework} support")
        init_framework.add_argument("--project-dir", default=".", help="Path to project root")

    observe_parser = sub.add_parser("observe", help="Observe repeated test runs")
    observe_sub = observe_parser.add_subparsers(dest="framework", required=True)
    for framework in supported_frameworks():
        observe_framework = observe_sub.add_parser(framework, help=f"Observe {framework} runs")
        observe_framework.add_argument("--project-dir", default=".", help="Path to project root")
        observe_framework.add_argument(
            "--runs",
            type=int,
            default=None,
            help=f"Number of repeated framework invocations (default: {DEFAULT_RUNS})",
        )
        observe_framework.add_argument(
            "--min-observations",
            type=int,
            default=None,
            help=f"Minimum observations required for non_flaky labels (default: {DEFAULT_MIN_OBSERVATIONS})",
        )
        observe_framework.add_argument(
            "--reuse-history",
            action="append",
            default=[],
            help="Manifest file or directory to import into the managed corpus",
        )
        _add_framework_args(observe_framework, framework)

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
    for framework in supported_frameworks():
        predict_framework = predict_sub.add_parser(framework, help=f"Predict {framework} flakiness")
        predict_framework.add_argument("--project-dir", default=".", help="Path to project root")
        predict_framework.add_argument(
            "--format",
            choices=["table", "json"],
            default="table",
            help="Output format",
        )
        predict_framework.add_argument("--output", help="Optional output path for rendered report")
        _add_framework_args(predict_framework, framework)

    diagnostics_parser = sub.add_parser(
        "diagnostics",
        help="Emit editor-oriented diagnostics for the current suite",
    )
    diagnostics_sub = diagnostics_parser.add_subparsers(dest="framework", required=True)
    for framework in supported_frameworks():
        diagnostics_framework = diagnostics_sub.add_parser(
            framework,
            help=f"Emit {framework} diagnostics",
        )
        diagnostics_framework.add_argument("--project-dir", default=".", help="Path to project root")
        diagnostics_framework.add_argument(
            "--format",
            choices=["json"],
            default="json",
            help="Output format",
        )
        diagnostics_framework.add_argument("--output", help="Optional output path for rendered diagnostics")
        _add_framework_args(diagnostics_framework, framework)

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


def _adapter_for(framework: str) -> FrameworkAdapter:
    return get_adapter(framework)


def _default_from_config(project_root: Path, section: str, key: str, fallback):
    config = load_config(project_root)
    return config.get(section, {}).get(key, fallback)


def _load_workspace_manifests(project_root: Path) -> List:
    return [load_manifest(path) for path in list_workspace_manifests(project_root)]


def _run_framework_observation(
    *,
    adapter: FrameworkAdapter,
    project_root: Path,
    framework_args: Sequence[str],
) -> Path:
    run_id = str(uuid.uuid4())
    manifest_path = run_manifest_path(project_root, run_id)
    command = adapter.build_observe_invocation(
        project_root=project_root,
        manifest_path=manifest_path,
        framework_args=framework_args,
    )
    returncode, stdout, stderr = adapter.run(project_root=project_root, command=command)
    if returncode not in (0, 1, 5):
        raise RuntimeError(
            f"{adapter.framework} observation failed with exit code {returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    if not manifest_path.exists():
        raise RuntimeError(f"{adapter.framework} observation finished without producing a manifest.")
    return manifest_path


def _run_framework_collection(
    *,
    adapter: FrameworkAdapter,
    project_root: Path,
    framework_args: Sequence[str],
) -> Path:
    run_id = str(uuid.uuid4())
    manifest_path = run_manifest_path(project_root, run_id)
    command = adapter.build_collect_invocation(
        project_root=project_root,
        manifest_path=manifest_path,
        framework_args=framework_args,
    )
    returncode, stdout, stderr = adapter.run(project_root=project_root, command=command)
    if returncode not in (0, 5):
        raise RuntimeError(
            f"{adapter.framework} collection failed with exit code {returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    if not manifest_path.exists():
        raise RuntimeError(f"{adapter.framework} collection finished without producing a manifest.")
    return manifest_path


def run_init(args: argparse.Namespace) -> int:
    project_root = _project_root(args.project_dir)
    paths = ensure_workspace(project_root)
    print(f"Initialized FLAST workspace at {paths.workspace_dir}")
    print(f"{args.framework} integration is wrapper-driven; no framework config changes were made.")
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
        _run_framework_observation(
            adapter=adapter,
            project_root=project_root,
            framework_args=args.framework_args,
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

    feature_texts = []
    for label in labels:
        adapter = _adapter_for(label.framework)
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
        framework_args=args.framework_args,
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
    framework_args: Sequence[str],
) -> PredictionReport:
    ensure_workspace(project_root)
    adapter = _adapter_for(framework)
    manifest_path = _run_framework_collection(
        adapter=adapter,
        project_root=project_root,
        framework_args=framework_args,
    )
    manifest = adapter.load_manifest(manifest_path)
    artifacts = load_trained_artifacts(project_root)
    feature_inputs = [
        adapter.resolve_feature_input(item.test_id, Path(manifest.project_root))
        for item in manifest.collected
    ]
    feature_texts = [build_feature_text(item) for item in feature_inputs]
    predictions = predict_with_explanations(artifacts, feature_texts)
    labels_by_test = {
        label.test_id: label
        for label in load_labels(ensure_workspace(project_root).labels_path)
    }
    records = []
    for feature, prediction in zip(feature_inputs, predictions):
        explanation = _build_prediction_explanation(
            feature=feature,
            prediction=prediction,
            observed_label=labels_by_test.get(feature.test_id)
            or labels_by_test.get(feature.normalized_test_id),
        )
        record = PredictionRecord(
            test_id=feature.test_id,
            score=prediction.score,
            predicted_label=prediction.predicted_label,
            class_name=feature.class_name,
            test_name=feature.test_name,
            file_path=str(feature.file_path) if feature.file_path else None,
            explanation=explanation,
        )
        records.append(
            PredictionRecord(
                test_id=record.test_id,
                score=record.score,
                predicted_label=record.predicted_label,
                class_name=record.class_name,
                test_name=record.test_name,
                file_path=record.file_path,
                line=adapter.resolve_prediction_line(record, Path(manifest.project_root)),
                explanation=record.explanation,
            )
        )
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


def _build_prediction_explanation(
    *,
    feature: FeatureInput,
    prediction: ExplainedPrediction,
    observed_label: Optional[LabelRecord],
) -> dict:
    static_signals = detect_static_signals(feature)
    return {
        "summary": _prediction_summary(
            prediction=prediction,
            static_signals=static_signals,
            observed_label=observed_label,
        ),
        "observed_history": _observed_history(observed_label),
        "nearest_examples": [item.to_dict() for item in prediction.nearest_examples],
        "static_signals": static_signals,
    }


def _observed_history(label: Optional[LabelRecord]) -> Optional[dict]:
    if label is None:
        return None
    return {
        "label": label.label,
        "reason": label.reason,
        "total_observations": label.total_observations,
        "outcome_counts": dict(label.outcome_counts),
    }


def _prediction_summary(
    *,
    prediction: ExplainedPrediction,
    static_signals: Sequence[dict],
    observed_label: Optional[LabelRecord],
) -> str:
    parts: List[str] = []
    flaky_neighbors = [
        item for item in prediction.nearest_examples if item.label == "flaky"
    ]
    non_flaky_neighbors = [
        item for item in prediction.nearest_examples if item.label == "non_flaky"
    ]
    if observed_label and observed_label.label == "flaky":
        parts.append(
            "previous observations for this test include both passing and failure-like outcomes"
        )
    elif flaky_neighbors:
        parts.append(
            "similar to {count} observed flaky test{suffix}".format(
                count=len(flaky_neighbors),
                suffix="" if len(flaky_neighbors) == 1 else "s",
            )
        )
    elif non_flaky_neighbors:
        parts.append("nearest examples are mostly observed non-flaky tests")

    if static_signals:
        parts.append(
            "source signals include "
            + ", ".join(signal["kind"] for signal in static_signals[:3])
        )

    if not parts:
        parts.append("model score is based on nearest labeled tests")
    return "; ".join(parts)


def run_diagnostics(args: argparse.Namespace) -> int:
    report = create_prediction_report(
        project_root=_project_root(args.project_dir),
        framework=args.framework,
        framework_args=args.framework_args,
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
