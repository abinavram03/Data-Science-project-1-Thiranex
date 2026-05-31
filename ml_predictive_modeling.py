"""
Predictive Modeling Using Machine Learning
==========================================
Applies Linear Regression, Decision Trees, and Random Forest to
built-in datasets. Trains/tests models and visualizes performance
via confusion matrices, ROC curves, and learning curves.

Requirements:
    pip install scikit-learn matplotlib seaborn numpy pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from itertools import cycle


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "blue":   "#3266AD",
    "green":  "#3B6D11",
    "orange": "#E07B39",
    "purple": "#6B52B7",
    "red":    "#C0392B",
    "gray":   "#7F8C8D",
    "light":  "#F4F6F9",
    "dark":   "#2C3E50",
}
PALETTE   = [COLORS["blue"], COLORS["green"], COLORS["orange"],
             COLORS["purple"], COLORS["red"]]
ALGO_CLRS = {"Random Forest": COLORS["blue"],
             "Decision Tree": COLORS["orange"],
             "Logistic Regression": COLORS["green"]}


# ── Dataset loader ─────────────────────────────────────────────────────────────
def load_dataset(name: str):
    loaders = {"iris": load_iris, "breast_cancer": load_breast_cancer, "wine": load_wine}
    ds = loaders[name]()
    return ds.data, ds.target, ds.target_names, ds.feature_names, ds.DESCR.splitlines()[0]


# ── Model definitions ──────────────────────────────────────────────────────────
def build_models():
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
    }


# ── Training & evaluation ─────────────────────────────────────────────────────
def evaluate_model(model, X_train, X_test, y_train, y_test, class_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    avg = "binary" if len(class_names) == 2 else "macro"

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
        "f1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
        "cm":        confusion_matrix(y_test, y_pred),
        "report":    classification_report(y_test, y_pred,
                                           target_names=class_names, zero_division=0),
        "y_pred":    y_pred,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(ax, cm, class_names, title, color):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", ax=ax,
        cmap=sns.light_palette(color, as_cmap=True),
        linewidths=0.5, linecolor="#E0E0E0",
        xticklabels=class_names, yticklabels=class_names,
        cbar=False, annot_kws={"size": 11, "weight": "bold"}
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)


def plot_roc_curves(ax, models_fitted, X_test, y_test, class_names):
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    for name, model in models_fitted.items():
        color = ALGO_CLRS[name]
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            y_score = model.decision_function(X_test)
            if y_score.ndim == 1:
                y_score = np.column_stack([-y_score, y_score])

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_bin[:, 0] if y_bin.ndim > 1 else y_bin,
                                    y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f"{name} (AUC={roc_auc:.3f})")
        else:
            # micro-average for multiclass
            fpr_all, tpr_all, _ = roc_curve(y_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr_all, tpr_all)
            ax.plot(fpr_all, tpr_all, lw=2, color=color,
                    label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random (AUC=0.500)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — all models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)


def plot_learning_curve(ax, model, X, y, name, color):
    sizes, train_sc, val_sc = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5, scoring="accuracy", n_jobs=-1
    )
    train_mean = train_sc.mean(axis=1)
    train_std  = train_sc.std(axis=1)
    val_mean   = val_sc.mean(axis=1)
    val_std    = val_sc.std(axis=1)

    ax.plot(sizes, train_mean, "o--", color=COLORS["gray"], lw=2,
            markersize=5, label="Training score")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.12, color=COLORS["gray"])
    ax.plot(sizes, val_mean, "o-", color=color, lw=2,
            markersize=5, label="Validation score")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=color)
    ax.set_title(f"Learning Curve — {name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training samples", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim([0.55, 1.02])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def plot_feature_importance(ax, model, feature_names, name):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        idx = np.argsort(imp)[::-1][:10]
        bars = ax.barh(
            [feature_names[i] for i in idx[::-1]],
            imp[idx[::-1]],
            color=ALGO_CLRS.get(name, COLORS["blue"]),
            edgecolor="white", linewidth=0.5
        )
        ax.set_title(f"Feature Importance — {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(labelsize=9)
    else:
        ax.text(0.5, 0.5, f"Not available\nfor {name}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color=COLORS["gray"])
        ax.set_title(f"Feature Importance — {name}", fontsize=11, fontweight="bold")
        ax.axis("off")


def plot_metrics_comparison(ax, results):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    names   = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.22

    for i, name in enumerate(names):
        vals  = [results[name][m] for m in metrics]
        color = ALGO_CLRS.get(name, PALETTE[i])
        bars  = ax.bar(x + i * width, vals, width,
                       label=name, color=color,
                       edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison — all metrics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.25)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_pipeline(dataset_name: str = "iris", test_size: float = 0.2):
    print("=" * 65)
    print(f"  ML Predictive Modeling — dataset: {dataset_name.upper()}")
    print("=" * 65)

    # Load data
    X, y, class_names, feature_names, desc = load_dataset(dataset_name)
    print(f"\n{desc}")
    print(f"  Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  Classes: {len(class_names)}")
    print(f"  Classes: {', '.join(class_names)}\n")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train & evaluate all models
    models  = build_models()
    results = {}
    fitted  = {}

    for name, model in models.items():
        res = evaluate_model(model, X_train, X_test, y_train, y_test, class_names)
        results[name] = res
        fitted[name]  = model
        print(f"── {name} ──────────────────────────────────────────")
        print(f"  Accuracy : {res['accuracy']:.4f}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall   : {res['recall']:.4f}")
        print(f"  F1 Score : {res['f1']:.4f}")
        print(f"\n{res['report']}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(20, 22), facecolor="white")
    fig.suptitle(
        f"Machine Learning — Predictive Modeling\nDataset: {dataset_name.capitalize()}",
        fontsize=16, fontweight="bold", y=0.98, color=COLORS["dark"]
    )

    outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.42,
                              top=0.95, bottom=0.04)

    # Row 0: metric comparison + ROC
    row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.32)
    ax_metrics = fig.add_subplot(row0[0])
    ax_roc     = fig.add_subplot(row0[1])
    plot_metrics_comparison(ax_metrics, results)
    plot_roc_curves(ax_roc, fitted, X_test, y_test, class_names)

    # Row 1: confusion matrices
    row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.38)
    for i, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(row1[i])
        plot_confusion_matrix(ax, res["cm"], class_names,
                              f"Confusion Matrix\n{name}",
                              list(ALGO_CLRS.values())[i])

    # Row 2: learning curves
    row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.35)
    for i, (name, model) in enumerate(models.items()):
        ax = fig.add_subplot(row2[i])
        plot_learning_curve(ax, model, X, y, name, list(ALGO_CLRS.values())[i])

    # Row 3: feature importance
    row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.38)
    for i, (name, model) in enumerate(models.items()):
        ax = fig.add_subplot(row3[i])
        plot_feature_importance(ax, model, feature_names, name)

    plt.savefig(f"ml_report_{dataset_name}.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    print(f"\n✓  Report saved → ml_report_{dataset_name}.png")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Predictive Modeling")
    parser.add_argument(
        "--dataset", choices=["iris", "breast_cancer", "wine"],
        default="iris",
        help="Dataset to use (default: iris)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Test split fraction 0.1–0.4 (default: 0.2)"
    )
    args = parser.parse_args()
    run_pipeline(dataset_name=args.dataset, test_size=args.test_size)
