"""Next-day Charlotte high-temperature and rain models from loaddata.csv.

Run: python weather_models.py
Requires: pandas, numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAIN_PROBABILITY_CUTOFF = 0.15
TEST_START = pd.Timestamp("2026-01-01")
WINDOW_DAYS = 7


@dataclass
class TreeNode:
    probability: float
    feature: int = -1
    threshold: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Ordinary least squares with train-only scaling for numerical stability."""
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1
    design = np.column_stack((np.ones(len(x)), (x - means) / scales))
    standardized_coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    coefficients = standardized_coefficients[1:] / scales
    intercept = standardized_coefficients[0] - means @ coefficients
    return float(intercept), coefficients


def fit_decision_tree(
    x: np.ndarray, y: np.ndarray, max_depth: int = 4, min_leaf: int = 30
) -> tuple[TreeNode, np.ndarray]:
    """Fit a binary CART classification tree using Gini impurity."""
    importances = np.zeros(x.shape[1])

    def grow(indices: np.ndarray, depth: int) -> TreeNode:
        labels = y[indices]
        n = len(indices)
        node = TreeNode(float(labels.mean()))
        if depth >= max_depth or n < 2 * min_leaf or labels.min() == labels.max():
            return node

        parent_gini = 2 * node.probability * (1 - node.probability)
        best_gain, best_feature, best_threshold = 0.0, -1, 0.0
        for feature in range(x.shape[1]):
            values = x[indices, feature]
            order = np.argsort(values, kind="mergesort")
            ordered_values, ordered_labels = values[order], labels[order]
            positions = np.arange(min_leaf, n - min_leaf + 1)
            positions = positions[
                ordered_values[positions - 1] < ordered_values[positions]
            ]
            if not len(positions):
                continue
            left_pos = np.cumsum(ordered_labels)[positions - 1]
            right_pos = labels.sum() - left_pos
            weighted_gini = 2 * (
                left_pos * (positions - left_pos) / positions
                + right_pos * ((n - positions) - right_pos) / (n - positions)
            ) / n
            choice = int(np.argmin(weighted_gini))
            gain = float(parent_gini - weighted_gini[choice])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feature = feature
                split = int(positions[choice])
                best_threshold = float(
                    (ordered_values[split - 1] + ordered_values[split]) / 2
                )

        if best_feature < 0:
            return node
        left_mask = x[indices, best_feature] <= best_threshold
        node.feature, node.threshold = best_feature, best_threshold
        importances[best_feature] += best_gain * n / len(y)
        node.left = grow(indices[left_mask], depth + 1)
        node.right = grow(indices[~left_mask], depth + 1)
        return node

    root = grow(np.arange(len(y)), 0)
    if importances.sum():
        importances /= importances.sum()
    return root, importances


def tree_probabilities(root: TreeNode, x: np.ndarray) -> np.ndarray:
    probabilities = np.empty(len(x))
    for row_number, row in enumerate(x):
        node = root
        while node.feature >= 0:
            node = node.left if row[node.feature] <= node.threshold else node.right
        probabilities[row_number] = node.probability
    return probabilities


def load_daily_data(csv_path: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(csv_path, encoding="utf-8")
    # The degree symbol in the supplied CSV was already damaged on export.
    if not (
        len(raw.columns) == 8
        and raw.columns[0] == "Date"
        and raw.columns[1].startswith("Temp High")
        and raw.columns[2].startswith("Temp Low")
        and list(raw.columns[3:])
        == ["Dew Pt High/Low", "Humidity", "Max Wind", "Pressure", "Precip."]
    ):
        raise ValueError(f"Unexpected CSV columns: {list(raw.columns)}")

    raw = raw.rename(columns={
        "Date": "date", raw.columns[1]: "high_f",
        raw.columns[2]: "low_f", "Precip.": "precip_in",
    })
    raw["date"] = pd.to_datetime(raw["date"], format="%m/%d/%Y", errors="raise")
    if raw.loc[raw.duplicated("date", keep=False)].groupby("date").nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting records found for the same date")
    duplicate_count = int(raw.duplicated("date").sum())
    raw = raw.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    for name in ("high_f", "low_f"):
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    # Source values have text units such as "0.97 in".
    raw["precip_in"] = pd.to_numeric(
        raw["precip_in"].astype(str).str.extract(r"^\s*([0-9]*\.?[0-9]+)", expand=False),
        errors="coerce",
    )
    if (raw["precip_in"].dropna() < 0).any():
        raise ValueError("Negative precipitation found")
    return raw[["date", "high_f", "low_f", "precip_in"]], duplicate_count


def make_examples(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    examples = pd.DataFrame({"observation_date": daily["date"]})
    for field in ("high_f", "low_f", "precip_in"):
        for lag in range(WINDOW_DAYS):
            examples[f"{field}_lag_{lag}"] = daily[field].shift(lag)

    next_date = daily["date"].shift(-1)
    examples["target_date"] = next_date
    examples["target_high_f"] = daily["high_f"].shift(-1)
    next_precip = daily["precip_in"].shift(-1)
    examples["target_rain"] = (next_precip > 0).astype(int)
    examples.loc[next_precip.isna(), "target_rain"] = np.nan

    # Calendar date is known before the forecast and captures seasonality.
    day_of_year = next_date.dt.dayofyear
    examples["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    examples["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    feature_names = [c for c in examples if "_lag_" in c] + ["day_sin", "day_cos"]

    contiguous = (
        (daily["date"] - daily["date"].shift(WINDOW_DAYS - 1))
        == pd.Timedelta(days=WINDOW_DAYS - 1)
    ) & (next_date - daily["date"] == pd.Timedelta(days=1))
    examples = examples.loc[contiguous].dropna(
        subset=feature_names + ["target_high_f", "target_rain"]
    ).reset_index(drop=True)
    examples["target_rain"] = examples["target_rain"].astype(int)
    return examples, feature_names


def regression_metrics(y: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    actual = y.to_numpy()
    errors = actual - prediction
    return {
        "MAE (F)": float(np.mean(np.abs(errors))),
        "RMSE (F)": float(np.sqrt(np.mean(errors**2))),
        "R2": float(1 - np.sum(errors**2) / np.sum((actual - actual.mean())**2)),
    }


def classification_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    actual = y.to_numpy()
    prediction = (probability > RAIN_PROBABILITY_CUTOFF).astype(int)
    tp = np.sum((actual == 1) & (prediction == 1))
    fp = np.sum((actual == 0) & (prediction == 1))
    fn = np.sum((actual == 1) & (prediction == 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "Accuracy": float(np.mean(actual == prediction)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "Brier": float(np.mean((actual - probability) ** 2)),
    }


def markdown_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    columns = list(rows[0][1])
    lines = [
        "| Method | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    lines += [
        "| " + name + " | " + " | ".join(f"{values[c]:.3f}" for c in columns) + " |"
        for name, values in rows
    ]
    return "\n".join(lines)


def run(csv_path: Path) -> str:
    daily, duplicate_count = load_daily_data(csv_path)
    examples, features = make_examples(daily)
    train = examples.loc[examples["target_date"] < TEST_START]
    test = examples.loc[examples["target_date"] >= TEST_START]
    if train.empty or test.empty:
        raise ValueError("Need both pre-2026 training data and 2026 test data")

    x_train, x_test = train[features].to_numpy(), test[features].to_numpy()
    high_train, high_test = train["target_high_f"], test["target_high_f"]
    rain_train, rain_test = train["target_rain"], test["target_rain"]

    intercept, coefficients = fit_linear_regression(x_train, high_train.to_numpy())
    high_prediction = intercept + x_test @ coefficients
    high_baseline = test["high_f_lag_0"].to_numpy()  # tomorrow equals today's high

    # Fixed complexity avoids selecting a tree using the held-out 2026 results.
    tree, importances = fit_decision_tree(x_train, rain_train.to_numpy())
    rain_probability = tree_probabilities(tree, x_test)
    prevalence_probability = np.full(len(test), rain_train.mean())
    persistence_probability = (test["precip_in_lag_0"] > 0).astype(float).to_numpy()

    regression_rows = [
        ("Today's high (baseline)", regression_metrics(high_test, high_baseline)),
        ("Linear regression", regression_metrics(high_test, high_prediction)),
    ]
    classification_rows = [
        ("Training rain rate (baseline)", classification_metrics(rain_test, prevalence_probability)),
        ("Today's rain (baseline)", classification_metrics(rain_test, persistence_probability)),
        ("Decision tree", classification_metrics(rain_test, rain_probability)),
    ]
    rain_prediction = (rain_probability > RAIN_PROBABILITY_CUTOFF).astype(int)
    actual_rain = rain_test.to_numpy()
    tn = int(np.sum((actual_rain == 0) & (rain_prediction == 0)))
    fp = int(np.sum((actual_rain == 0) & (rain_prediction == 1)))
    fn = int(np.sum((actual_rain == 1) & (rain_prediction == 0)))
    tp = int(np.sum((actual_rain == 1) & (rain_prediction == 1)))

    top_coefficients = sorted(
        zip(features, coefficients), key=lambda pair: abs(pair[1]), reverse=True
    )[:8]
    top_importances = sorted(
        zip(features, importances), key=lambda pair: pair[1], reverse=True
    )[:8]
    lines = [
        "# Next-day Charlotte weather model results",
        "",
        "Generated by `python weather_models.py --output model_results.md` from the repository CSV.",
        "",
        f"- Source rows: {len(daily) + duplicate_count}; duplicate dates removed: {duplicate_count}.",
        f"- Date coverage after cleaning: {daily.date.min():%Y-%m-%d} to {daily.date.max():%Y-%m-%d}.",
        f"- Training examples: {len(train)} ({train.target_date.min():%Y-%m-%d} to {train.target_date.max():%Y-%m-%d}).",
        f"- Held-out examples: {len(test)} ({test.target_date.min():%Y-%m-%d} to {test.target_date.max():%Y-%m-%d}).",
        f"- Training rain rate: {rain_train.mean():.1%}; test rain rate: {rain_test.mean():.1%}.",
        "",
        "## Tomorrow's maximum temperature",
        "",
        markdown_table(regression_rows),
        "",
        "On the 2026 holdout, linear regression lowers MAE and RMSE relative to the persistence baseline.",
        "",
        "The persistence baseline predicts that tomorrow's high equals today's observed high.",
        "Lower MAE/RMSE and higher R2 are better.",
        "",
        "Largest absolute linear-regression coefficients (per one unit of each feature):",
        "",
        *[f"- `{name}`: {coefficient:+.3f}" for name, coefficient in top_coefficients],
        "",
        "## Rain tomorrow",
        "",
        "The observed event is next-day precipitation greater than 0 inches. The decision is rain when the model's predicted probability is **strictly greater than 15%**.",
        "",
        markdown_table(classification_rows),
        "",
        f"Decision-tree confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}.",
        "The 15% cutoff catches most rainy days but creates many false alarms; its accuracy and Brier score do not improve on the relevant baselines here.",
        "Brier score measures probability error; lower is better. At a 15% cutoff, recall may rise while false alarms also rise.",
        "Tree probabilities are uncalibrated leaf frequencies, so treat the 15% decision as a project rule rather than a calibrated weather-service forecast.",
        "",
        "Largest decision-tree feature importances:",
        "",
        *[f"- `{name}`: {importance:.3f}" for name, importance in top_importances],
        "",
        "## Validation notes",
        "",
        "Each row predicts the next calendar day from the previous seven observed days. No target-day weather values enter the features. The split uses target dates, so no 2026 outcome is used to fit either model. Model settings were fixed before looking at the held-out results.",
        "These are retrospective results on the supplied historical CSV, not a live forecast for the current date.",
        "",
        "The final course report still needs a third trained model, exploratory figures, literature, and societal analysis.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("loaddata.csv"))
    parser.add_argument("--output", type=Path, help="Optional Markdown results file")
    args = parser.parse_args()
    results = run(args.data)
    if args.output:
        args.output.write_text(results, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(results)


if __name__ == "__main__":
    main()
