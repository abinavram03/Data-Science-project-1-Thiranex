"""
Exploratory Data Analysis (EDA) Project
========================================
Performs comprehensive EDA on built-in datasets:
  - Statistical summaries (mean, median, std, skewness, kurtosis)
  - Distribution plots, box plots, violin plots
  - Correlation heatmaps and pair plots
  - Outlier detection (IQR method)
  - Key influencing factors via feature importance
  - Structured PDF/PNG report

Requirements:
    pip install scikit-learn matplotlib seaborn numpy pandas scipy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import argparse
import os

# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    "blue":   "#3266AD",
    "teal":   "#1D9E75",
    "amber":  "#E07B39",
    "purple": "#6B52B7",
    "red":    "#C0392B",
    "gray":   "#7F8C8D",
    "light":  "#F4F6F9",
    "dark":   "#2C3E50",
    "muted":  "#95A5A6",
}
PALETTE = [C["blue"], C["teal"], C["amber"], C["purple"], C["red"]]
sns.set_palette(PALETTE)


# ── Dataset loaders ────────────────────────────────────────────────────────────
def load_dataset(name: str):
    """Return (DataFrame, target_series, task_type, description)."""
    if name == "iris":
        ds = load_iris(as_frame=True)
        df = ds.data.copy()
        df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
        target = pd.Series(
            [ds.target_names[i] for i in ds.target], name="species"
        )
        return df, target, "classification", "Fisher's Iris Dataset", ds.target_names.tolist()

    if name == "wine":
        ds = load_wine(as_frame=True)
        df = ds.data.copy()
        target = pd.Series(
            [f"Class_{i}" for i in ds.target], name="wine_class"
        )
        return df, target, "classification", "Wine Recognition Dataset", [f"Class_{i}" for i in range(3)]

    if name == "breast_cancer":
        ds = load_breast_cancer(as_frame=True)
        df = ds.data.iloc[:, :10].copy()   # top 10 features for readability
        target = pd.Series(
            [ds.target_names[i] for i in ds.target], name="diagnosis"
        )
        return df, target, "classification", "Breast Cancer Wisconsin Dataset", ds.target_names.tolist()

    if name == "diabetes":
        ds = load_diabetes(as_frame=True)
        df = ds.data.copy()
        target = ds.target.rename("progression")
        return df, target, "regression", "Diabetes Progression Dataset", []

    raise ValueError(f"Unknown dataset: {name}")


# ── Statistical summary ────────────────────────────────────────────────────────
def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe().T
    desc["skewness"] = df.skew()
    desc["kurtosis"] = df.kurtosis()
    desc["missing"]  = df.isnull().sum()
    desc["missing%"] = (df.isnull().mean() * 100).round(2)
    return desc.round(4)


# ── Outlier detection (IQR) ────────────────────────────────────────────────────
def detect_outliers(df: pd.DataFrame) -> pd.Series:
    counts = {}
    for col in df.select_dtypes(include=np.number).columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        counts[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    return pd.Series(counts, name="outlier_count")


# ── Feature importance ─────────────────────────────────────────────────────────
def compute_importance(df: pd.DataFrame, target, task: str) -> pd.Series:
    le = LabelEncoder()
    y  = le.fit_transform(target) if task == "classification" else target.values
    X  = df.select_dtypes(include=np.number).fillna(df.median(numeric_only=True))

    if task == "classification":
        mdl = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        mdl = RandomForestRegressor(n_estimators=100, random_state=42)

    mdl.fit(X, y)
    return pd.Series(mdl.feature_importances_, index=X.columns, name="importance").sort_values(ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=C["dark"])
    ax.set_xlabel(xlabel, fontsize=9, color=C["gray"])
    ax.set_ylabel(ylabel, fontsize=9, color=C["gray"])
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


# ── 1. Distribution grid ───────────────────────────────────────────────────────
def plot_distributions(df, target, task, fig_path):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    n = len(num_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2),
                             facecolor="white")
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = df[col].dropna()

        if task == "classification":
            cats = target.unique()
            for j, cat in enumerate(sorted(cats)):
                mask = target == cat
                ax.hist(df.loc[mask, col].dropna(), bins=20,
                        alpha=0.65, color=PALETTE[j % len(PALETTE)],
                        label=str(cat), edgecolor="white", linewidth=0.4)
            ax.legend(fontsize=7, framealpha=0.7)
        else:
            ax.hist(data, bins=25, color=C["blue"], edgecolor="white",
                    linewidth=0.4, alpha=0.85)

        # KDE overlay
        try:
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde   = stats.gaussian_kde(data)
            ax2   = ax.twinx()
            ax2.plot(kde_x, kde(kde_x), color=C["dark"], lw=1.4, alpha=0.7)
            ax2.set_yticks([])
            ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
        except Exception:
            pass

        sk = data.skew()
        ax.set_title(f"{col}\nskew={sk:.2f}", fontsize=9, fontweight="bold",
                     color=C["dark"], pad=4)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold",
                 color=C["dark"], y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  Distribution plot → {fig_path}")


# ── 2. Box + Violin plots ─────────────────────────────────────────────────────
def plot_boxviolin(df, target, task, fig_path):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()[:8]
    n = len(num_cols)
    fig, axes = plt.subplots(2, n, figsize=(max(14, n * 2), 8), facecolor="white")

    for i, col in enumerate(num_cols):
        # Box plot
        ax_box = axes[0][i]
        if task == "classification":
            cats = sorted(target.unique())
            data_by_cat = [df.loc[target == c, col].dropna().values for c in cats]
            bp = ax_box.boxplot(data_by_cat, patch_artist=True, widths=0.55,
                                medianprops={"color": "white", "linewidth": 2})
            for patch, clr in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(clr)
                patch.set_alpha(0.8)
            ax_box.set_xticks(range(1, len(cats) + 1))
            ax_box.set_xticklabels([str(c)[:8] for c in cats], fontsize=7)
        else:
            ax_box.boxplot(df[col].dropna(), patch_artist=True,
                           boxprops={"facecolor": C["blue"], "alpha": 0.7},
                           medianprops={"color": "white", "linewidth": 2})

        ax_box.set_title(col[:14], fontsize=8, fontweight="bold", color=C["dark"])
        ax_box.spines[["top", "right"]].set_visible(False)
        ax_box.tick_params(labelsize=7)
        if i == 0:
            ax_box.set_ylabel("Box Plot", fontsize=8, color=C["gray"])

        # Violin plot
        ax_vio = axes[1][i]
        if task == "classification":
            tmp = pd.DataFrame({col: df[col], "target": target})
            sns.violinplot(x="target", y=col, data=tmp, ax=ax_vio,
                           palette=PALETTE[:len(cats)], inner="quartile",
                           linewidth=0.8, cut=0)
            ax_vio.set_xticklabels([str(c)[:8] for c in sorted(target.unique())],
                                   fontsize=7)
            ax_vio.set_xlabel("")
        else:
            sns.violinplot(y=df[col], ax=ax_vio,
                           color=C["teal"], inner="quartile",
                           linewidth=0.8, cut=0)

        ax_vio.set_title("", fontsize=8)
        ax_vio.spines[["top", "right"]].set_visible(False)
        ax_vio.tick_params(labelsize=7)
        if i == 0:
            ax_vio.set_ylabel("Violin Plot", fontsize=8, color=C["gray"])

    fig.suptitle("Box Plots & Violin Plots", fontsize=14, fontweight="bold",
                 color=C["dark"])
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  Box/violin plot  → {fig_path}")


# ── 3. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation(df, fig_path):
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

    # Full heatmap
    ax = axes[0]
    sns.heatmap(
        corr, annot=True, fmt=".2f", ax=ax,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        center=0, linewidths=0.4, linecolor="#E0E0E0",
        annot_kws={"size": 7}, square=True, cbar_kws={"shrink": 0.75}
    )
    ax.set_title("Correlation Matrix", fontsize=12, fontweight="bold", color=C["dark"])
    ax.tick_params(labelsize=8)

    # Top correlated pairs bar chart
    ax2 = axes[1]
    pairs = (
        corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
            .stack()
            .reset_index()
    )
    pairs.columns = ["f1", "f2", "corr"]
    pairs["abs"] = pairs["corr"].abs()
    pairs = pairs.nlargest(12, "abs")
    pairs["label"] = pairs["f1"].str[:10] + " × " + pairs["f2"].str[:10]

    colors_bar = [C["blue"] if v >= 0 else C["red"] for v in pairs["corr"]]
    bars = ax2.barh(pairs["label"], pairs["corr"], color=colors_bar,
                    edgecolor="white", linewidth=0.5, height=0.65)
    ax2.axvline(0, color=C["dark"], linewidth=0.8)
    ax2.set_title("Top 12 Correlated Pairs", fontsize=12, fontweight="bold", color=C["dark"])
    ax2.set_xlabel("Pearson r", fontsize=9, color=C["gray"])
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.25)

    for bar, v in zip(bars, pairs["corr"]):
        ax2.text(v + (0.01 if v >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center",
                 ha="left" if v >= 0 else "right",
                 fontsize=7.5, fontweight="bold",
                 color=C["dark"])

    pos_patch = mpatches.Patch(color=C["blue"], label="Positive")
    neg_patch = mpatches.Patch(color=C["red"],  label="Negative")
    ax2.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  Correlation plot → {fig_path}")


# ── 4. Pair plot ───────────────────────────────────────────────────────────────
def plot_pairplot(df, target, task, fig_path):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()[:5]
    tmp = df[num_cols].copy()
    tmp["_target"] = target.astype(str)

    hue = "_target" if task == "classification" else None
    palette = {str(k): PALETTE[i % len(PALETTE)]
               for i, k in enumerate(sorted(tmp["_target"].unique()))} if hue else None

    g = sns.pairplot(
        tmp, hue=hue, palette=palette,
        diag_kind="kde", plot_kws={"alpha": 0.55, "s": 20, "edgecolor": "none"},
        diag_kws={"fill": True, "linewidth": 1.2}
    )
    g.figure.suptitle("Pair Plot — Feature Relationships", y=1.01,
                       fontsize=13, fontweight="bold", color=C["dark"])
    g.figure.set_facecolor("white")

    plt.savefig(fig_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  Pair plot        → {fig_path}")


# ── 5. Feature importance + outlier summary ────────────────────────────────────
def plot_importance_outliers(importance, outliers, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    # Feature importance
    ax = axes[0]
    top = importance.head(10)
    colors_imp = [PALETTE[i % len(PALETTE)] for i in range(len(top))]
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=colors_imp[::-1], edgecolor="white",
                   linewidth=0.5, height=0.65)
    for bar, v in zip(bars, top.values[::-1]):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8, fontweight="bold", color=C["dark"])
    _style_ax(ax, "Feature Importance (Random Forest)", "Importance Score", "Feature")
    ax.set_xlim(0, top.values.max() * 1.2)

    # Outlier counts
    ax2 = axes[1]
    out_sorted = outliers.sort_values(ascending=False)
    clrs = [C["red"] if v > 5 else C["amber"] if v > 2 else C["teal"]
            for v in out_sorted.values]
    ax2.bar(out_sorted.index, out_sorted.values, color=clrs,
            edgecolor="white", linewidth=0.5)
    ax2.set_xticklabels(out_sorted.index, rotation=40, ha="right", fontsize=8)
    _style_ax(ax2, "Outlier Count per Feature (IQR Method)",
              "Feature", "Number of Outliers")

    red_p   = mpatches.Patch(color=C["red"],   label="> 5 outliers")
    amber_p = mpatches.Patch(color=C["amber"], label="3–5 outliers")
    teal_p  = mpatches.Patch(color=C["teal"],  label="≤ 2 outliers")
    ax2.legend(handles=[red_p, amber_p, teal_p], fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  Importance/outlier plot → {fig_path}")


# ── 6. Structured text report ─────────────────────────────────────────────────
def print_report(name, desc, df, target, task, summary, outliers, importance, class_names):
    sep   = "═" * 65
    sep2  = "─" * 65

    print(f"\n{sep}")
    print(f"  EDA REPORT  ·  {desc}")
    print(sep)

    # 1. Dataset overview
    print("\n【 1. DATASET OVERVIEW 】")
    print(f"  Rows        : {df.shape[0]}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Task type   : {task.capitalize()}")
    if task == "classification":
        vc = target.value_counts()
        print(f"  Classes     : {', '.join(str(c) for c in vc.index)}")
        print(f"  Class dist  : {dict(vc)}")
    else:
        print(f"  Target range: {target.min():.2f} – {target.max():.2f}  "
              f"(mean={target.mean():.2f})")
    print(f"  Missing vals: {df.isnull().sum().sum()} total")

    # 2. Statistical summary
    print(f"\n【 2. STATISTICAL SUMMARY 】")
    print(summary[["mean", "std", "min", "50%", "max", "skewness", "kurtosis", "missing"]]
          .to_string())

    # 3. Skewness insights
    print(f"\n【 3. DISTRIBUTION INSIGHTS 】")
    skewed = summary[summary["skewness"].abs() > 1]
    if len(skewed):
        print("  Highly skewed features (|skew| > 1) — consider log transform:")
        for col, row in skewed.iterrows():
            print(f"    • {col:<30}  skew = {row['skewness']:+.3f}")
    else:
        print("  No heavily skewed features detected.")

    # 4. Correlations
    print(f"\n【 4. CORRELATION INSIGHTS 】")
    corr = df.corr(numeric_only=True)
    pairs = (
        corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
            .stack()
            .reset_index()
    )
    pairs.columns = ["f1", "f2", "corr"]
    strong_pos = pairs[pairs["corr"] >  0.7].sort_values("corr", ascending=False)
    strong_neg = pairs[pairs["corr"] < -0.7].sort_values("corr")

    if len(strong_pos):
        print("  Strong positive correlations (r > 0.7):")
        for _, row in strong_pos.iterrows():
            print(f"    • {row['f1']:<25} ↔ {row['f2']:<25}  r = {row['corr']:+.3f}")
    if len(strong_neg):
        print("  Strong negative correlations (r < -0.7):")
        for _, row in strong_neg.iterrows():
            print(f"    • {row['f1']:<25} ↔ {row['f2']:<25}  r = {row['corr']:+.3f}")
    if not len(strong_pos) and not len(strong_neg):
        print("  No strong correlations (|r| > 0.7) found.")

    # 5. Outliers
    print(f"\n【 5. OUTLIER SUMMARY (IQR Method) 】")
    for feat, cnt in outliers.sort_values(ascending=False).items():
        bar   = "█" * min(cnt, 30)
        label = "⚠ HIGH" if cnt > 5 else ("△ MED" if cnt > 2 else "✓")
        print(f"  {feat:<30} {cnt:>3} outliers  {bar}  {label}")

    # 6. Feature importance
    print(f"\n【 6. KEY INFLUENCING FEATURES 】")
    for rank, (feat, score) in enumerate(importance.head(5).items(), 1):
        bar = "█" * int(score * 50)
        print(f"  #{rank}  {feat:<30}  {score:.4f}  {bar}")

    # 7. Recommendations
    print(f"\n【 7. RECOMMENDATIONS 】")
    recs = []
    if df.isnull().sum().sum() > 0:
        recs.append("Handle missing values via imputation or removal.")
    if len(skewed) > 0:
        recs.append("Apply log/sqrt transforms to skewed features before modelling.")
    if len(strong_pos) + len(strong_neg) > 2:
        recs.append("Consider removing highly correlated features to reduce multicollinearity.")
    high_out = outliers[outliers > 5]
    if len(high_out):
        recs.append(f"Investigate/cap outliers in: {', '.join(high_out.index[:3])}.")
    recs.append(f"Top predictive features: {', '.join(importance.head(3).index)}.")
    recs.append("Scale features (StandardScaler) before distance-based models.")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(dataset_name: str = "iris", output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"eda_{dataset_name}")

    print(f"\n{'='*65}")
    print(f"  EDA Pipeline — {dataset_name.upper()}")
    print(f"{'='*65}")

    # Load
    df, target, task, desc, class_names = load_dataset(dataset_name)
    print(f"\n  Loaded: {desc}  ({df.shape[0]} rows × {df.shape[1]} cols)\n")

    # Compute stats
    summary    = statistical_summary(df)
    outliers   = detect_outliers(df)
    importance = compute_importance(df, target, task)

    # Print structured report
    print_report(dataset_name, desc, df, target, task,
                 summary, outliers, importance, class_names)

    # Generate plots
    print("  Generating plots …\n")
    plot_distributions  (df, target, task, f"{prefix}_1_distributions.png")
    plot_boxviolin      (df, target, task, f"{prefix}_2_boxviolin.png")
    plot_correlation    (df,               f"{prefix}_3_correlation.png")
    plot_pairplot       (df, target, task, f"{prefix}_4_pairplot.png")
    plot_importance_outliers(importance, outliers, f"{prefix}_5_importance_outliers.png")

    print(f"\n  All plots saved to: {output_dir}/")
    print("  Done.\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA Project")
    parser.add_argument(
        "--dataset",
        choices=["iris", "wine", "breast_cancer", "diabetes"],
        default="iris",
        help="Dataset to analyse (default: iris)"
    )
    parser.add_argument(
        "--output",
        default="eda_output",
        help="Folder to save plot images (default: eda_output)"
    )
    args = parser.parse_args()
    run_eda(dataset_name=args.dataset, output_dir=args.output)
