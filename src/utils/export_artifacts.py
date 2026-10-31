"""Read-only checks for dashboard artifacts against saved Member 2 notebook output.

This module intentionally does not create evaluation values or train models.
Run it with ``python -m src.utils.export_artifacts`` to check whether the
integrated CSV/JSON artifacts agree with the outputs saved in the notebook.
"""

from html.parser import HTMLParser
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks" / "Model_Training_Rating_Classification.ipynb"
ARTIFACT_DIR = ROOT / "models"
NOTEBOOK_ARTIFACT_DIR = ROOT / "notebooks" / "models"


def _artifact_path(filename: str) -> Path:
    notebook_export = NOTEBOOK_ARTIFACT_DIR / filename
    return notebook_export if notebook_export.exists() else ARTIFACT_DIR / filename


class _FirstTableParser(HTMLParser):
    """Small stdlib HTML table reader for notebook's saved DataFrame output."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if not self.tables:
                self._table_depth = 1
            else:
                self._table_depth += 1
        elif self._table_depth == 1 and tag == "tr":
            self._row = []
        elif self._table_depth == 1 and tag in {"th", "td"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth == 1 and tag in {"th", "td"} and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif self._table_depth == 1 and tag == "tr" and self._row is not None:
            self.tables.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1


def _notebook_table(cell_index: int) -> pd.DataFrame:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell = notebook["cells"][cell_index]
    for output in cell.get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if html:
            parser = _FirstTableParser()
            parser.feed("".join(html))
            if parser.tables:
                rows = parser.tables
                header_index = next((i for i, row in enumerate(rows) if "model" in row or "target_app" in row), None)
                if header_index is None:
                    continue
                headers = rows[header_index]
                table = pd.DataFrame(rows[header_index + 1 :], columns=headers)
                table = table.loc[:, ~table.columns.astype(str).str.startswith("Unnamed")]
                for column in table.columns:
                    try:
                        table[column] = pd.to_numeric(table[column], errors="raise")
                    except (TypeError, ValueError):
                        pass
                return table
    raise ValueError(f"No saved HTML table found in notebook cell {cell_index}.")


def _compare_table(csv_name: str, notebook_cell: int, keys: list[str]) -> list[str]:
    path = _artifact_path(csv_name)
    if not path.exists():
        return [f"Missing {path.relative_to(ROOT)}"]
    saved = pd.read_csv(path)
    source = _notebook_table(notebook_cell)
    problems = []
    for key in keys:
        if key not in saved or key not in source:
            problems.append(f"{csv_name}: required column {key!r} missing")
            continue
        left = saved[key]
        right = source[key]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            if len(left) != len(right) or not np.allclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                rtol=0,
                atol=1e-6,
                equal_nan=True,
            ):
                problems.append(f"{csv_name}: values in {key!r} differ from notebook cell {notebook_cell}")
        elif left.astype(str).tolist() != right.astype(str).tolist():
            problems.append(f"{csv_name}: values in {key!r} differ from notebook cell {notebook_cell}")
    return problems


def validate_evaluation_artifacts() -> list[str]:
    """Check saved benchmark tables and confusion matrices without modifying them."""
    problems = []
    problems += _compare_table(
        "holdout_model_comparison.csv",
        18,
        ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"],
    )
    problems += _compare_table(
        "loao_results.csv",
        20,
        [
            "target_app", "train_rows", "test_rows", "accuracy", "macro_precision",
            "macro_recall", "macro_f1", "pure_test_rows_n_apps_eq_1",
            "pure_macro_f1_n_apps_eq_1",
        ],
    )

    holdout_path = _artifact_path("holdout_model_comparison.csv")
    if holdout_path.exists():
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        training_cell = notebook["cells"][16]
        training_source = "".join(training_cell.get("source", []))
        training_output = "\n".join(
            "".join(output.get("text", [])) for output in training_cell.get("outputs", [])
        )
        expected_configs = {
            "MultinomialNB": ("alpha=0.5", "sample_weight=balanced"),
            "RandomForest": ("n_estimators=150", "class_weight=balanced_subsample"),
            "LinearSVM_SGD": ("loss=hinge (SGD)", "class_weight=balanced"),
        }
        time_patterns = {
            "MultinomialNB": r"Naive Bayes selesai dalam ([0-9.]+) detik",
            "LinearSVM_SGD": r"Linear SVM selesai dalam ([0-9.]+) detik",
            "RandomForest": r"Random Forest selesai dalam ([0-9.]+) detik",
        }
        saved_holdout = pd.read_csv(holdout_path)
        for _, row in saved_holdout.iterrows():
            name = row["model"]
            if name not in expected_configs:
                continue
            expected_loss, expected_balance = expected_configs[name]
            if "loss_or_alpha" in row and str(row["loss_or_alpha"]) != expected_loss:
                problems.append(f"{name}: hyperparameter label differs from notebook cell 16")
            if "balancing" in row and str(row["balancing"]) != expected_balance:
                problems.append(f"{name}: balancing label differs from notebook cell 16")
            pattern = time_patterns[name]
            match = re.search(pattern, training_output)
            if "training_time_sec" in row and match and not np.isclose(
                float(row["training_time_sec"]), float(match.group(1)), rtol=0, atol=0.05
            ):
                problems.append(f"{name}: training time differs from notebook cell 16 output")
            if "training_time_sec" in row and not match:
                problems.append(f"{name}: training time source missing from saved notebook output")
        config_fragments = [
            "MultinomialNB(alpha=0.5)",
            'compute_sample_weight(class_weight="balanced", y=y_train)',
            'n_estimators=150',
            'loss="hinge"',
            'class_weight="balanced"',
            'class_weight="balanced_subsample"',
        ]
        for fragment in config_fragments:
            if fragment not in training_source:
                problems.append(f"Training configuration source not found in notebook cell 16: {fragment}")

    cm_path = _artifact_path("confusion_matrices.json")
    if not cm_path.exists():
        problems.append(f"Missing {cm_path.relative_to(ROOT)}")
        return problems
    matrices = json.loads(cm_path.read_text(encoding="utf-8"))
    holdout_path = _artifact_path("holdout_model_comparison.csv")
    if not holdout_path.exists():
        return problems
    holdout = pd.read_csv(holdout_path)
    labels = matrices.get("labels", [])
    if not labels:
        return problems + ["Confusion-matrix labels are missing."]
    for _, row in holdout.iterrows():
        name = row["model"]
        if name not in matrices:
            problems.append(f"No confusion matrix for {name}")
            continue
        matrix = np.asarray(matrices[name], dtype=np.float64)
        if matrix.shape != (len(labels), len(labels)):
            problems.append(f"Invalid confusion-matrix shape for {name}: {matrix.shape}")
            continue
        tp = np.diag(matrix)
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
        observed = {
            "accuracy": tp.sum() / matrix.sum(),
            "macro_precision": precision.mean(),
            "macro_recall": recall.mean(),
            "macro_f1": f1.mean(),
            "weighted_f1": np.average(f1, weights=support),
        }
        for metric, value in observed.items():
            if metric in row and not np.isclose(float(row[metric]), value, rtol=0, atol=1e-6):
                problems.append(f"{name}: {metric} from confusion matrix differs from holdout CSV")
    return problems


if __name__ == "__main__":
    issues = validate_evaluation_artifacts()
    if issues:
        print("Artifact validation found issues:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Evaluation artifacts match the saved notebook tables and confusion-matrix metrics.")
