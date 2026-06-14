"""
performance_evaluator.py
------------------------
Module A, Component 1 — AI Reliability and Performance Test Suite
AI Assurance Toolkit | U.S. Public-Sector Edition

PURPOSE:
    Evaluates a trained machine learning model's performance against a labeled
    test dataset. Produces quantitative metrics used to assess whether a model
    meets the reliability threshold required for operational deployment in a
    government or regulated-sector context.

FEDERAL ALIGNMENT:
    Satisfies the NIST AI Risk Management Framework (AI RMF, 2023) — MEASURE
    function, specifically MR-2.5: "AI system performance or assurance criteria
    are established" and MR-2.6: "Evaluations are conducted on AI system
    performance." Also supports OMB Memorandum M-25-21 documentation
    requirements for deployment readiness reviews.

INPUTS:
    - A trained, scikit-learn-compatible classification or regression model
      (loaded from a .pkl or .joblib file)
    - A CSV test dataset with features and a labeled target column

OUTPUTS:
    - Console-printed metrics summary
    - module_a_outputs/performance_report.json  (structured, human-readable)
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "module_a_outputs"
OUTPUT_FILENAME = "performance_report.json"


# ---------------------------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    class_labels: list,
) -> dict:
    """
    Compute the full suite of classification performance metrics.

    Each metric is explained inline so that developers and reviewers unfamiliar
    with a given statistic can understand its operational significance.

    Args:
        y_true:       Ground-truth labels from the test dataset.
        y_pred:       Predicted class labels produced by the model.
        y_prob:       Predicted class probabilities (required for AUC-ROC and
                      calibration). Pass None if the model does not support
                      probability outputs.
        class_labels: Ordered list of unique class label values.

    Returns:
        Dictionary of metric names to computed values (floats or dicts).
    """
    metrics = {}

    # ------------------------------------------------------------------
    # ACCURACY
    # What it measures: The percentage of all predictions the model got right.
    # Why it matters:   Provides a single top-line number for overall correctness.
    #                   However, it can be misleading when class sizes are unequal
    #                   (e.g., 95% of records belong to one class).
    # ------------------------------------------------------------------
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Determine whether this is binary or multi-class for averaging strategy
    is_binary = len(class_labels) == 2
    avg_strategy = "binary" if is_binary else "weighted"

    # ------------------------------------------------------------------
    # PRECISION
    # What it measures: Of all cases the model flagged as positive, what
    #                   fraction were actually positive?
    # Why it matters:   Low precision means many false alarms. In government
    #                   contexts (e.g., benefits eligibility), false alarms
    #                   can impose unnecessary burden on individuals.
    # ------------------------------------------------------------------
    metrics["precision"] = float(
        precision_score(y_true, y_pred, average=avg_strategy, zero_division=0)
    )

    # ------------------------------------------------------------------
    # RECALL (also called Sensitivity or True Positive Rate)
    # What it measures: Of all actual positive cases, what fraction did the
    #                   model correctly identify?
    # Why it matters:   Low recall means the model is missing real cases. In
    #                   high-stakes settings (e.g., fraud detection, safety
    #                   screening), missed detections can be costly or dangerous.
    # ------------------------------------------------------------------
    metrics["recall"] = float(
        recall_score(y_true, y_pred, average=avg_strategy, zero_division=0)
    )

    # ------------------------------------------------------------------
    # F1 SCORE
    # What it measures: The harmonic mean of precision and recall. It balances
    #                   both false alarms and missed detections into a single number.
    # Why it matters:   Useful when both types of error are important. A high F1
    #                   score indicates the model handles both precision and recall
    #                   well simultaneously.
    # ------------------------------------------------------------------
    metrics["f1_score"] = float(
        f1_score(y_true, y_pred, average=avg_strategy, zero_division=0)
    )

    # ------------------------------------------------------------------
    # FALSE POSITIVE RATE (FPR)
    # What it measures: Of all actual negative cases, what fraction did the
    #                   model incorrectly flag as positive?
    # Why it matters:   High FPR leads to resources being spent investigating
    #                   non-issues. Critical in screening or triage contexts.
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    if is_binary:
        # For binary: cm = [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    else:
        # For multi-class: compute macro-averaged FPR and FNR
        n_classes = len(class_labels)
        fpr_list, fnr_list = [], []
        for i in range(n_classes):
            tp_i = cm[i, i]
            fn_i = cm[i, :].sum() - tp_i
            fp_i = cm[:, i].sum() - tp_i
            tn_i = cm.sum() - tp_i - fn_i - fp_i
            fpr_list.append(fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0.0)
            fnr_list.append(fn_i / (fn_i + tp_i) if (fn_i + tp_i) > 0 else 0.0)
        fpr = float(np.mean(fpr_list))
        fnr = float(np.mean(fnr_list))

    # ------------------------------------------------------------------
    # FALSE NEGATIVE RATE (FNR)
    # What it measures: Of all actual positive cases, what fraction did the
    #                   model miss (classify as negative)?
    # Why it matters:   Directly related to recall (FNR = 1 - Recall). Especially
    #                   critical in safety, health, or fraud contexts where missing
    #                   a real event has severe consequences.
    # ------------------------------------------------------------------
    metrics["false_positive_rate"] = fpr
    metrics["false_negative_rate"] = fnr

    # ------------------------------------------------------------------
    # AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
    # What it measures: The model's ability to distinguish between classes across
    #                   all possible decision thresholds. Ranges from 0.5 (random
    #                   guessing) to 1.0 (perfect discrimination).
    # Why it matters:   Unlike accuracy, AUC-ROC is threshold-independent. A value
    #                   above 0.80 is generally considered good; below 0.70 raises
    #                   concerns about model reliability.
    # ------------------------------------------------------------------
    if y_prob is not None:
        try:
            if is_binary:
                # Use the probability of the positive class (column index 1)
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                # Multi-class: one-vs-rest macro-averaged AUC
                y_true_bin = label_binarize(y_true, classes=class_labels)
                auc = float(
                    roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted")
                )
            metrics["auc_roc"] = auc
        except Exception as exc:
            metrics["auc_roc"] = None
            metrics["auc_roc_error"] = str(exc)
    else:
        metrics["auc_roc"] = None
        metrics["auc_roc_note"] = "Model does not support probability outputs; AUC-ROC not computed."

    # ------------------------------------------------------------------
    # CALIBRATION SCORE (Brier Score)
    # What it measures: How closely the model's predicted probabilities match
    #                   actual observed outcomes. A score of 0.0 is perfect;
    #                   0.25 is equivalent to always predicting 50% probability.
    # Why it matters:   A well-calibrated model is important when predicted
    #                   probabilities are used for decision-making thresholds
    #                   (e.g., "flag if probability > 0.7"). Poor calibration
    #                   means confidence scores cannot be trusted at face value.
    # ------------------------------------------------------------------
    if y_prob is not None and is_binary:
        try:
            brier = float(brier_score_loss(y_true, y_prob[:, 1]))
            metrics["calibration_brier_score"] = brier
        except Exception as exc:
            metrics["calibration_brier_score"] = None
            metrics["calibration_brier_error"] = str(exc)
    else:
        metrics["calibration_brier_score"] = None
        if y_prob is None:
            metrics["calibration_note"] = "Calibration requires probability outputs."
        else:
            metrics["calibration_note"] = "Brier score computed for binary classification only."

    # ------------------------------------------------------------------
    # PER-CLASS BREAKDOWN
    # What it measures: Precision, recall, and F1 for each individual class.
    # Why it matters:   Overall metrics can hide poor performance on minority
    #                   classes. Per-class detail is required for fairness review.
    # ------------------------------------------------------------------
    report_dict = classification_report(
        y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )
    metrics["per_class_metrics"] = {
        str(label): {
            "precision": round(report_dict[str(label)]["precision"], 4),
            "recall": round(report_dict[str(label)]["recall"], 4),
            "f1_score": round(report_dict[str(label)]["f1-score"], 4),
            "support": int(report_dict[str(label)]["support"]),
        }
        for label in class_labels
        if str(label) in report_dict
    }

    return metrics


# ---------------------------------------------------------------------------
# PLAIN-ENGLISH INTERPRETATION
# ---------------------------------------------------------------------------

def generate_plain_english_summary(metrics: dict, model_name: str) -> dict:
    """
    Translate computed metrics into plain-English findings and operational
    implications for a non-technical government program manager audience.

    Args:
        metrics:    The metrics dictionary from compute_classification_metrics().
        model_name: Human-readable name of the model being evaluated.

    Returns:
        Dictionary with 'findings', 'concerns', 'strengths', and
        'deployment_signal' keys.
    """
    findings = []
    concerns = []
    strengths = []

    accuracy = metrics.get("accuracy")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    f1 = metrics.get("f1_score")
    fpr = metrics.get("false_positive_rate")
    fnr = metrics.get("false_negative_rate")
    auc = metrics.get("auc_roc")
    brier = metrics.get("calibration_brier_score")

    # --- Accuracy ---
    if accuracy is not None:
        pct = round(accuracy * 100, 1)
        findings.append(
            f"Overall Accuracy: The model correctly predicted the outcome in {pct}% of "
            f"test cases. "
            + (
                "This is generally considered strong baseline performance."
                if accuracy >= 0.85
                else "This level of accuracy warrants careful review before operational deployment."
                if accuracy >= 0.70
                else "This accuracy level is low and raises significant concerns about model reliability."
            )
        )
        if accuracy >= 0.85:
            strengths.append(f"High overall accuracy ({pct}%).")
        elif accuracy < 0.70:
            concerns.append(f"Overall accuracy of {pct}% is below the recommended 70% threshold for deployment consideration.")

    # --- Precision ---
    if precision is not None:
        pct = round(precision * 100, 1)
        findings.append(
            f"Precision: When the model predicts a positive outcome, it is correct {pct}% of the time. "
            + (
                "This indicates a low rate of false alarms."
                if precision >= 0.80
                else "This suggests a meaningful rate of false alarms that may affect operational trust."
            )
        )
        if precision < 0.70:
            concerns.append(f"Precision of {pct}% means more than 30% of positive predictions are incorrect (false alarms).")

    # --- Recall ---
    if recall is not None:
        pct = round(recall * 100, 1)
        findings.append(
            f"Recall (Detection Rate): The model correctly identified {pct}% of actual positive cases. "
            + (
                "Few real cases are being missed."
                if recall >= 0.80
                else "A notable portion of real cases are being missed, which may be a safety or mission concern."
            )
        )
        if recall < 0.70:
            concerns.append(f"Recall of {pct}% means more than 30% of actual positive cases are going undetected.")

    # --- F1 Score ---
    if f1 is not None:
        pct = round(f1 * 100, 1)
        findings.append(
            f"F1 Score (Balanced Performance): The combined precision-recall balance score is {pct}%. "
            + (
                "The model handles both false alarms and missed detections well."
                if f1 >= 0.80
                else "The model shows meaningful trade-offs between false alarms and missed detections."
            )
        )
        if f1 >= 0.80:
            strengths.append(f"Strong F1 score ({pct}%) indicating balanced performance.")

    # --- False Positive Rate ---
    if fpr is not None:
        pct = round(fpr * 100, 1)
        if fpr > 0.15:
            concerns.append(
                f"False Positive Rate of {pct}%: The model incorrectly flags {pct}% of non-cases as positive. "
                "This may place undue burden on individuals or resources."
            )
        else:
            strengths.append(f"Low false positive rate ({pct}%).")

    # --- False Negative Rate ---
    if fnr is not None:
        pct = round(fnr * 100, 1)
        if fnr > 0.15:
            concerns.append(
                f"False Negative Rate of {pct}%: The model misses {pct}% of actual positive cases. "
                "This could result in undetected issues with operational or mission impact."
            )

    # --- AUC-ROC ---
    if auc is not None:
        findings.append(
            f"Discrimination Ability (AUC-ROC): The model's ability to distinguish between outcomes "
            f"scores {round(auc, 3)} on a scale of 0.5 (random chance) to 1.0 (perfect). "
            + (
                "This indicates strong discriminative power."
                if auc >= 0.80
                else "This indicates moderate discriminative power; further review is advised."
                if auc >= 0.70
                else "This score is close to random chance, raising serious questions about model validity."
            )
        )
        if auc >= 0.80:
            strengths.append(f"Strong AUC-ROC score ({round(auc, 3)}).")
        elif auc < 0.70:
            concerns.append(f"AUC-ROC of {round(auc, 3)} is near random chance; model may lack meaningful predictive power.")

    # --- Calibration ---
    if brier is not None:
        findings.append(
            f"Calibration (Brier Score): The model's confidence scores are calibrated with a Brier score "
            f"of {round(brier, 4)} (lower is better; 0.25 = random guessing). "
            + (
                "Confidence scores appear reliable."
                if brier <= 0.10
                else "Confidence scores should be interpreted with caution."
                if brier <= 0.20
                else "Confidence scores are poorly calibrated and should not be used for threshold-based decisions."
            )
        )
        if brier > 0.20:
            concerns.append(f"High Brier score ({round(brier, 4)}) indicates unreliable probability estimates.")

    # --- Overall deployment signal ---
    n_concerns = len(concerns)
    if n_concerns == 0:
        deployment_signal = "APPROVED FOR DEPLOYMENT"
        signal_explanation = (
            "Performance metrics are strong across all dimensions. No significant concerns were identified. "
            "The model appears suitable for deployment pending subgroup and robustness review."
        )
    elif n_concerns <= 2:
        deployment_signal = "APPROVED WITH CONDITIONS"
        signal_explanation = (
            f"{n_concerns} performance concern(s) were identified. The model may be deployable with "
            "additional monitoring, human oversight, or restricted scope. Review the concerns listed below."
        )
    else:
        deployment_signal = "NOT RECOMMENDED FOR DEPLOYMENT"
        signal_explanation = (
            f"{n_concerns} performance concerns were identified. The model does not appear ready for "
            "operational deployment without significant remediation. A detailed remediation plan should be "
            "developed before re-evaluation."
        )

    return {
        "deployment_signal": deployment_signal,
        "signal_explanation": signal_explanation,
        "findings": findings,
        "strengths": strengths,
        "concerns": concerns,
        "note": (
            "This summary is based solely on overall test-set performance. "
            "Subgroup disparity analysis (Component 2) and robustness testing (Component 3) "
            "are required before a final deployment recommendation can be issued."
        ),
    }


# ---------------------------------------------------------------------------
# REPORT ASSEMBLY AND OUTPUT
# ---------------------------------------------------------------------------

def assemble_report(
    metrics: dict,
    plain_english: dict,
    model_name: str,
    dataset_path: str,
    n_samples: int,
    class_labels: list,
    output_dir: str,
) -> dict:
    """
    Assemble all computed data into the structured JSON report dictionary
    and write it to disk.

    Args:
        metrics:       Computed metric values.
        plain_english: Plain-English interpretations.
        model_name:    Name of the evaluated model.
        dataset_path:  Path to the test dataset (for traceability).
        n_samples:     Number of test records evaluated.
        class_labels:  List of class label values.
        output_dir:    Directory where the JSON file will be saved.

    Returns:
        The fully assembled report as a Python dictionary.
    """
    report = {
        "report_metadata": {
            "report_type": "Model Performance Evaluation",
            "toolkit": "AI Assurance Toolkit — Module A",
            "component": "Component 1: Model Performance Evaluator",
            "federal_alignment": [
                "NIST AI RMF (2023) — MEASURE function, MR-2.5, MR-2.6: Quantifying AI system performance",
                "OMB Memorandum M-25-21 — Documentation supporting deployment readiness reviews",
                "America's AI Action Plan (July 2025) — Responsible AI deployment evaluation",
            ],
            "model_name": model_name,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_dataset": str(dataset_path),
            "test_sample_size": n_samples,
            "class_labels": [str(c) for c in class_labels],
        },
        "performance_metrics": {
            "overall": {
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1_score"], 4),
                "false_positive_rate": round(metrics["false_positive_rate"], 4),
                "false_negative_rate": round(metrics["false_negative_rate"], 4),
                "auc_roc": round(metrics["auc_roc"], 4) if metrics.get("auc_roc") is not None else None,
                "calibration_brier_score": (
                    round(metrics["calibration_brier_score"], 4)
                    if metrics.get("calibration_brier_score") is not None
                    else None
                ),
            },
            "per_class": metrics.get("per_class_metrics", {}),
        },
        "plain_english_summary": plain_english,
    }

    # Propagate any notes/errors for optional metrics
    for key in ("auc_roc_note", "auc_roc_error", "calibration_note", "calibration_brier_error"):
        if key in metrics:
            report["performance_metrics"]["overall"][key] = metrics[key]

    # Write to disk
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


# ---------------------------------------------------------------------------
# CONSOLE OUTPUT
# ---------------------------------------------------------------------------

def print_console_summary(report: dict) -> None:
    """Print a formatted summary of key findings to the console."""
    meta = report["report_metadata"]
    overall = report["performance_metrics"]["overall"]
    summary = report["plain_english_summary"]

    divider = "=" * 70
    print(f"\n{divider}")
    print("  AI ASSURANCE TOOLKIT — Module A, Component 1")
    print("  Model Performance Evaluation Report")
    print(divider)
    print(f"  Model:           {meta['model_name']}")
    print(f"  Evaluation Date: {meta['evaluation_date']}")
    print(f"  Test Samples:    {meta['test_sample_size']:,}")
    print(f"  Classes:         {', '.join(meta['class_labels'])}")
    print(divider)
    print("  PERFORMANCE METRICS")
    print(f"    Accuracy:              {overall['accuracy']:.4f}")
    print(f"    Precision:             {overall['precision']:.4f}")
    print(f"    Recall:                {overall['recall']:.4f}")
    print(f"    F1 Score:              {overall['f1_score']:.4f}")
    print(f"    False Positive Rate:   {overall['false_positive_rate']:.4f}")
    print(f"    False Negative Rate:   {overall['false_negative_rate']:.4f}")
    auc_val = overall.get("auc_roc")
    print(f"    AUC-ROC:               {f'{auc_val:.4f}' if auc_val is not None else 'N/A'}")
    brier_val = overall.get("calibration_brier_score")
    print(f"    Calibration (Brier):   {f'{brier_val:.4f}' if brier_val is not None else 'N/A'}")
    print(divider)
    print(f"  DEPLOYMENT SIGNAL:  >>> {summary['deployment_signal']} <<<")
    print(f"\n  {summary['signal_explanation']}")
    if summary["concerns"]:
        print("\n  CONCERNS IDENTIFIED:")
        for i, c in enumerate(summary["concerns"], 1):
            print(f"    {i}. {c}")
    if summary["strengths"]:
        print("\n  STRENGTHS IDENTIFIED:")
        for s in summary["strengths"]:
            print(f"    ✓ {s}")
    print(divider)
    print(f"  Report saved to: {DEFAULT_OUTPUT_DIR}/{OUTPUT_FILENAME}")
    print(f"{divider}\n")


# ---------------------------------------------------------------------------
# PUBLIC API — callable from orchestrator
# ---------------------------------------------------------------------------

def run_performance_evaluation(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Unnamed Model",
    dataset_path: str = "N/A",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Primary entry point for programmatic use (called by run_evaluation.py).

    Args:
        model:        A fitted scikit-learn-compatible classifier.
        X_test:       Feature matrix for the test set (pandas DataFrame).
        y_test:       True labels for the test set (pandas Series).
        model_name:   Human-readable name for the report.
        dataset_path: Original dataset file path (for audit traceability).
        output_dir:   Folder where output files will be written.

    Returns:
        The assembled report dictionary.
    """
    class_labels = sorted(y_test.unique().tolist())
    y_pred = model.predict(X_test)

    # Attempt to retrieve probability estimates
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None
    else:
        y_prob = None

    metrics = compute_classification_metrics(y_test.values, y_pred, y_prob, class_labels)
    plain_english = generate_plain_english_summary(metrics, model_name)
    report = assemble_report(
        metrics, plain_english, model_name, dataset_path,
        len(y_test), class_labels, output_dir
    )
    print_console_summary(report)
    return report


def evaluate_from_files(
    model_path: str,
    dataset_path: str,
    target: str,
    model_name: str = "Unnamed Model",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Load a trained model and test dataset from files, then run performance evaluation.
    This function is used by the command-line interface.
    """
    validate_inputs(model_path, dataset_path)

    print(f"\n[INFO] Loading model from:   {model_path}")
    model = joblib.load(model_path)

    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    y_test = df[target]

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        missing_features = [col for col in expected_features if col not in df.columns]

        if missing_features:
            raise ValueError(
                f"The dataset is missing required model feature columns: {missing_features}"
            )

        X_test = df[expected_features]
    else:
        X_test = df.drop(columns=[target])

    print(f"[INFO] Dataset loaded: {len(df):,} rows, {len(X_test.columns)} features.")
    print(f"[INFO] Beginning performance evaluation for: {model_name}\n")

    return run_performance_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
    )
   
# ---------------------------------------------------------------------------
# STANDALONE CLI ENTRYPOINT
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Component 1 — Model Performance Evaluator (AI Assurance Toolkit)"
    )
    parser.add_argument("--model", required=True, help="Path to the serialized model file (.pkl or .joblib)")
    parser.add_argument("--dataset", required=True, help="Path to the test dataset CSV file")
    parser.add_argument("--target", required=True, help="Name of the target (label) column in the dataset")
    parser.add_argument("--model-name", default="Unnamed Model", help="Human-readable model name for the report")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output files")
    return parser.parse_args()


def validate_inputs(model_path: str, dataset_path: str) -> None:
    """
    Validate that required input files exist before any processing begins.
    Exits with a descriptive error message if validation fails.
    """
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model file not found: '{model_path}'")
        print("        Please verify the file path and try again.")
        sys.exit(1)
    if not Path(dataset_path).exists():
        print(f"\n[ERROR] Dataset file not found: '{dataset_path}'")
        print("        Please verify the file path and try again.")
        sys.exit(1)


def main() -> None:
    """Standalone CLI entry point."""
    args = parse_args()
    validate_inputs(args.model, args.dataset)

    print(f"\n[INFO] Loading model from:   {args.model}")
    model = joblib.load(args.model)

    print(f"[INFO] Loading dataset from: {args.dataset}")
    df = pd.read_csv(args.dataset)

    if args.target not in df.columns:
        print(f"\n[ERROR] Target column '{args.target}' not found in dataset.")
        print(f"        Available columns: {list(df.columns)}")
        sys.exit(1)

    X_test = df.drop(columns=[args.target])
    y_test = df[args.target]

    print(f"[INFO] Dataset loaded: {len(df):,} rows, {len(X_test.columns)} features.")
    print(f"[INFO] Beginning performance evaluation for: {args.model_name}\n")

    run_performance_evaluation(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=args.model_name,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
