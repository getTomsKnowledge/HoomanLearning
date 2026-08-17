"""
Name: results.py
Author: OpenAI Codex
Date: 08/16/2026
Description: The XOR binary classification task runtime.  Be sure to remember the name of your trained network!

Training result reporting utilities.

The functions here are intentionally data-driven: trainers pass plain dictionaries
that describe the network, run, losses, predictions, and reproducibility notes.
That keeps the report writer reusable for future network types.
"""

from pathlib import Path
import re

import numpy as np

from utils import cite_sources

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "runtimes" / "results"


def _to_python(value):
    """Convert NumPy and Path values into printable, JSON-like Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def _as_loss_list(training_loss):
    if training_loss is None:
        return []
    if isinstance(training_loss, np.ndarray):
        return [float(item) for item in training_loss.tolist()]
    return [float(item) for item in training_loss]


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _safe_filename_stem(name):
    stem = Path(str(name)).stem.strip()
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    return stem or "training_run"


def _report_name_from_result(result):
    reproducibility = result.get("reproducibility", {})
    saved_weights = reproducibility.get("saved_weights")

    if saved_weights and "<" not in str(saved_weights):
        return _safe_filename_stem(saved_weights)

    network = result.get("network", {})
    return _safe_filename_stem(network.get("name", "training_run"))


def summarize_parameters(feature_dict):
    """Return shape/count metadata for learned parameter arrays."""
    summary = {}
    total = 0

    for name, value in feature_dict.items():
        array = np.asarray(value)
        count = int(array.size)
        total += count
        summary[name] = {
            "shape": tuple(int(dim) for dim in array.shape),
            "count": count,
            "dtype": str(array.dtype),
        }

    summary["total_parameters"] = total
    return summary


def make_prediction_records(inputs, targets, predictions):
    """Package prediction rows for consistent result printing."""
    records = []
    for x_value, target_value, pred_value in zip(inputs, targets, predictions):
        pred_array = np.asarray(pred_value)
        records.append(
            {
                "input": _to_python(x_value),
                "target": _to_python(target_value),
                "prediction": _to_python(pred_array),
                "rounded": _to_python(np.rint(pred_array).astype(int)),
            }
        )
    return records


def build_training_results(
    network_name,
    network_kind,
    task,
    dataset,
    architecture,
    hyperparameters,
    feature_dict,
    training_loss,
    epochs_run,
    predictions=None,
    optimizer="gradient descent",
    loss_function="0.5 * sum squared error",
    initialization=None,
    saved_weights=None,
    notes=None,
):
    """Build a standard result dictionary from a completed training run."""
    loss_values = _as_loss_list(training_loss)
    final_loss = loss_values[-1] if loss_values else None
    cutoff = hyperparameters.get("loss_cutoff")
    converged = final_loss is not None and cutoff is not None and final_loss < cutoff

    return {
        "network": {
            "name": network_name,
            "kind": network_kind,
            "task": task,
            "architecture": _to_python(architecture),
        },
        "dataset": _to_python(dataset),
        "training": {
            "optimizer": optimizer,
            "loss_function": loss_function,
            "epochs_run": int(epochs_run),
            "max_epochs": int(hyperparameters.get("epochs", epochs_run)),
            "final_loss": final_loss,
            "loss_cutoff": cutoff,
            "converged": converged,
            "loss_history_length": len(loss_values),
        },
        "hyperparameters": _to_python(hyperparameters),
        "parameters": summarize_parameters(feature_dict),
        "predictions": predictions or [],
        "reproducibility": {
            "initialization": initialization or "See trainer and nn_helpers.initialize_features().",
            "saved_weights": _to_python(saved_weights) if saved_weights else "Saved interactively by nn.save_network() under networks/<name>.json.",
            "project_root": str(PROJECT_ROOT),
        },
        "notes": notes or [],
    }


def _append_mapping(lines, mapping, indent=0):
    prefix = "  " * indent
    for key, value in mapping.items():
        label = str(key).replace("_", " ").title()
        if isinstance(value, dict):
            lines.append(f"{prefix}- **{label}:**")
            _append_mapping(lines, value, indent + 1)
        else:
            lines.append(f"{prefix}- **{label}:** `{_format_value(value)}`")


def _append_section(lines, title, mapping):
    lines.append(f"## {title}")
    _append_mapping(lines, mapping)
    lines.append("")


def render_training_results_markdown(result, source_list=None):
    """Render a reproducible, network-aware training report as Markdown."""
    network = result.get("network", {})
    report_name = _report_name_from_result(result)
    lines = [
        f"# {report_name} Results",
        "",
        "## Summary",
        f"- **Network:** `{network.get('name')}`",
        f"- **Kind:** `{network.get('kind')}`",
        f"- **Task:** `{network.get('task')}`",
        "",
    ]

    _append_section(lines, "Architecture", network.get("architecture", {}))
    _append_section(lines, "Dataset", result.get("dataset", {}))
    _append_section(lines, "Training", result.get("training", {}))
    _append_section(lines, "Hyperparameters", result.get("hyperparameters", {}))
    _append_section(lines, "Parameters", result.get("parameters", {}))

    predictions = result.get("predictions", [])
    if predictions:
        lines.extend(
            [
                "## Final Predictions",
                "",
                "| Input | Target | Prediction | Rounded |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in predictions:
            lines.append(
                f"| `{row['input']}` | `{row['target']}` | "
                f"`{row['prediction']}` | `{row['rounded']}` |"
            )
        lines.append("")

    _append_section(lines, "Reproducibility", result.get("reproducibility", {}))

    notes = result.get("notes", [])
    if notes:
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    if source_list:
        lines.append("## IEEE Reference List")
        for source in sorted(source_list, key=lambda item: item.get("index", 0)):
            lines.append(f"- {cite_sources.generate_ieee_source(source)}")
        lines.append("")

    return "\n".join(lines)


def write_training_results(result, source_list=None, output_dir=None):
    """Write the training report to runtimes/results/<network>_results.md."""
    destination_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)

    report_name = _report_name_from_result(result)
    output_path = destination_dir / f"{report_name}_results.md"
    output_path.write_text(render_training_results_markdown(result, source_list), encoding="utf-8")
    return output_path


def print_training_results(result, source_list=None):
    """Compatibility wrapper: write the training report to Markdown."""
    output_path = write_training_results(result, source_list)
    print(f"Training results written to {output_path}")
    return output_path