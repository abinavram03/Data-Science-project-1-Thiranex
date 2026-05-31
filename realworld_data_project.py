"""
Real-World Data Science Project
================================
Three domain-specific pipelines — Finance, Health, Retail.
Each performs end-to-end analysis + prediction + visualisation.

    python realworld_data_project.py --domain finance
    python realworld_data_project.py --domain health
    python realworld_data_project.py --domain retail

Requirements:
    pip install scikit-learn matplotlib seaborn numpy pandas scipy
"""

import warnings, os, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
from sklearn.decomposition import PCA

# ── Global palette ─────────────────────────────────────────────────────────────
C = dict(blue="#3266AD", teal="#1D9E75", amber="#E07B39",
         red="#C0392B", purple="#6B52B7", gray="#7F8C8D",
         dark="#2C3E50", light="#F4F6F9", green="#27AE60")
PAL = [C["blue"], C["teal"], C["amber"], C["purple"], C["red"], C["green"]]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "font.size":        9,
})


# ══════════════════════════════════════════════════════════════════════════════
# ██  SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def make_stock_data(n_days=756, seed=42):
    """5 tech-stock-like price series with realistic random-walk."""
    rng = np.random.default_rng(seed)
    tickers = ["TECH", "FINX", "HLTH", "RETL", "ENRG"]
    starts  = [150.0, 85.0, 210.0, 60.0, 45.0]
    vols    = [0.018, 0.022, 0.015, 0.025, 0.030]
    drifts  = [0.0004, 0.0002, 0.0005, 0.0001, -0.0001]

    dates = pd.bdate_range("2021-01-04", periods=n_days)
    data  = {}
    for t, s, v, d in zip(tickers, starts, vols, drifts):
        log_ret = rng.normal(d, v, n_days)
        prices  = s * np.exp(np.cumsum(log_ret))
        data[t] = prices

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


def make_health_data(n=1200, seed=42):
    """Synthetic patient records — predict 30-day readmission (binary)."""
    rng = np.random.default_rng(seed)
    age            = rng.integers(18, 90, n)
    bmi            = np.clip(rng.normal(27, 5, n), 15, 55)
    blood_pressure = np.clip(rng.normal(125, 18, n), 70, 200)
    glucose        = np.clip(rng.normal(100, 25, n), 60, 300)
    hba1c          = np.clip(rng.normal(5.8, 1.2, n), 4, 12)
    num_meds       = rng.integers(0, 12, n)
    prev_admits    = rng.integers(0, 6, n)
    los            = rng.integers(1, 21, n)   # length of stay (days)
    gender         = rng.choice(["Male", "Female"], n)
    diagnosis      = rng.choice(["Diabetes", "Cardiac", "Respiratory",
                                  "Orthopedic", "Other"], n,
                                 p=[0.25, 0.22, 0.18, 0.15, 0.20])

    log_odds = (
        -4.5
        + 0.025 * (age - 50)
        + 0.04  * (bmi - 27)
        + 0.01  * (glucose - 100)
        + 0.30  * prev_admits
        + 0.08  * num_meds
        - 0.05  * los
        + rng.normal(0, 0.4, n)
    )
    prob      = 1 / (1 + np.exp(-log_odds))
    readmit   = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(dict(
        age=age, bmi=bmi, blood_pressure=blood_pressure,
        glucose=glucose, hba1c=hba1c, num_medications=num_meds,
        prev_admissions=prev_admits, length_of_stay=los,
        gender=gender, diagnosis=diagnosis, readmitted=readmit
    ))


def make_retail_data(n=2000, seed=42):
    """Synthetic retail transactions — predict customer CLV (regression)."""
    rng = np.random.default_rng(seed)
    tenure_months  = rng.integers(1, 72, n)
    num_orders     = rng.integers(1, 80, n)
    avg_order_val  = np.clip(rng.normal(85, 40, n), 5, 500)
    return_rate    = np.clip(rng.beta(2, 8, n), 0, 0.6)
    email_open     = np.clip(rng.beta(3, 5, n), 0, 1)
    discount_usage = np.clip(rng.beta(2, 6, n), 0, 1)
    category       = rng.choice(["Electronics","Apparel","Home","Sports","Beauty"], n,
                                 p=[0.22, 0.25, 0.20, 0.18, 0.15])
    region         = rng.choice(["North","South","East","West"], n)
    channel        = rng.choice(["Online","In-Store","Mobile"], n,
                                 p=[0.45, 0.30, 0.25])

    clv = (
        tenure_months * 1.5
        + num_orders  * avg_order_val * 0.08
        - return_rate * 300
        + email_open  * 120
        + rng.normal(0, 30, n)
    ).clip(10)

    segment = pd.cut(clv,
                     bins=[0, 150, 350, 600, np.inf],
                     labels=["Bronze","Silver","Gold","Platinum"])

    return pd.DataFrame(dict(
        tenure_months=tenure_months, num_orders=num_orders,
        avg_order_value=avg_order_val, return_rate=return_rate,
        email_open_rate=email_open, discount_usage=discount_usage,
        category=category, region=region, channel=channel,
        clv=clv.round(2), segment=segment
    ))


# ══════════════════════════════════════════════════════════════════════════════
# ██  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {os.path.basename(path)}")


def _title(ax, t, fs=11):
    ax.set_title(t, fontsize=fs, fontweight="bold", color=C["dark"], pad=8)


def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# ██  DOMAIN 1 — FINANCE
# ══════════════════════════════════════════════════════════════════════════════

def run_finance(out):
    section("FINANCE  ·  Stock Price Analysis & Return Prediction")

    df = make_stock_data()
    tickers = df.columns.tolist()

    # ── Features ──────────────────────────────────────────────────────────────
    returns = df.pct_change().dropna()
    rolling = df.rolling(20)
    ma20    = rolling.mean()
    vol20   = returns.rolling(20).std() * np.sqrt(252)

    # ── Stats ─────────────────────────────────────────────────────────────────
    ann_ret  = (returns.mean() * 252 * 100).round(2)
    ann_vol  = (returns.std()  * np.sqrt(252) * 100).round(2)
    sharpe   = (ann_ret / ann_vol).round(3)
    max_dd   = {}
    for t in tickers:
        roll_max   = df[t].cummax()
        drawdown   = (df[t] - roll_max) / roll_max
        max_dd[t]  = round(drawdown.min() * 100, 2)

    summary = pd.DataFrame({
        "Ann. Return %": ann_ret,
        "Ann. Volatility %": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": pd.Series(max_dd),
    })
    print("\n  Portfolio Summary")
    print(summary.to_string())

    # ── Predict next-day return (Ridge regression on lagged features) ─────────
    target_ticker = "TECH"
    feat_df = pd.DataFrame({
        "ret_1d":   returns[target_ticker].shift(1),
        "ret_5d":   returns[target_ticker].rolling(5).mean().shift(1),
        "vol_20d":  vol20[target_ticker].shift(1),
        "ma_diff":  ((df[target_ticker] - ma20[target_ticker]) / ma20[target_ticker]).shift(1),
    }).dropna()
    y = returns[target_ticker].loc[feat_df.index]

    Xtr, Xte, ytr, yte = train_test_split(feat_df, y, test_size=0.2, shuffle=False)
    mdl = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    mdl.fit(Xtr, ytr)
    ypred = mdl.predict(Xte)
    rmse  = np.sqrt(mean_squared_error(yte, ypred))
    r2    = r2_score(yte, ypred)
    print(f"\n  Return Prediction ({target_ticker})  RMSE={rmse:.5f}  R²={r2:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)
    fig.suptitle("Finance Dashboard — Stock Price Analysis",
                 fontsize=16, fontweight="bold", color=C["dark"], y=0.98)

    # 1. Price series
    ax = fig.add_subplot(gs[0, :])
    for i, t in enumerate(tickers):
        norm = df[t] / df[t].iloc[0] * 100
        ax.plot(df.index, norm, color=PAL[i], lw=1.6, label=t)
    ax.axhline(100, color=C["gray"], lw=0.8, ls="--")
    _title(ax, "Normalised Price Performance (Base=100)")
    ax.set_ylabel("Index (Day-1 = 100)")
    ax.legend(fontsize=9)

    # 2. Return heatmap (monthly)
    ax2 = fig.add_subplot(gs[1, 0])
    monthly = returns[tickers[0]].resample("ME").sum()
    pivot   = pd.DataFrame({
        "Month": monthly.index.strftime("%b-%y"),
        "Return": monthly.values
    }).set_index("Month")
    sns.heatmap(
        pivot.T, annot=True, fmt=".2%", ax=ax2,
        cmap=sns.diverging_palette(10, 130, as_cmap=True), center=0,
        linewidths=0.4, cbar=False, annot_kws={"size": 7}
    )
    _title(ax2, f"Monthly Returns Heatmap — {tickers[0]}")

    # 3. Volatility (rolling 20-day)
    ax3 = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(tickers):
        ax3.plot(vol20.index, vol20[t] * 100, color=PAL[i], lw=1.4, label=t, alpha=0.85)
    _title(ax3, "Rolling 20-Day Annualised Volatility (%)")
    ax3.set_ylabel("Volatility (%)")
    ax3.legend(fontsize=8)

    # 4. Correlation matrix
    ax4 = fig.add_subplot(gs[2, 0])
    corr = returns.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax4,
                cmap="Blues", linewidths=0.4, square=True,
                annot_kws={"size": 9}, cbar=False)
    _title(ax4, "Return Correlation Matrix")

    # 5. Sharpe / drawdown bar
    ax5 = fig.add_subplot(gs[2, 1])
    x = np.arange(len(tickers))
    w = 0.38
    b1 = ax5.bar(x - w/2, summary["Sharpe Ratio"],  w, color=C["blue"],  label="Sharpe Ratio",  alpha=0.88)
    b2 = ax5.bar(x + w/2, summary["Max Drawdown %"], w, color=C["red"],   label="Max Drawdown %", alpha=0.75)
    ax5.set_xticks(x); ax5.set_xticklabels(tickers)
    ax5.axhline(0, color=C["dark"], lw=0.8)
    ax5.legend(fontsize=8)
    _title(ax5, "Sharpe Ratio vs Max Drawdown")

    # 6. Predicted vs actual returns
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(yte.values,  color=C["blue"],  lw=1.4, label="Actual",    alpha=0.9)
    ax6.plot(ypred,        color=C["amber"], lw=1.4, label="Predicted", alpha=0.85, ls="--")
    _title(ax6, f"Predicted vs Actual Daily Returns — {target_ticker}")
    ax6.set_ylabel("Daily Return"); ax6.legend(fontsize=8)

    # 7. Drawdown chart
    ax7 = fig.add_subplot(gs[3, 1])
    for i, t in enumerate(tickers):
        rm  = df[t].cummax()
        dd  = (df[t] - rm) / rm * 100
        ax7.fill_between(df.index, dd, 0, alpha=0.35, color=PAL[i], label=t)
    _title(ax7, "Drawdown Chart (%)")
    ax7.set_ylabel("Drawdown (%)"); ax7.legend(fontsize=8)

    _save(fig, os.path.join(out, "finance_dashboard.png"))

    # Conclusions
    section("FINANCE — KEY CONCLUSIONS")
    best_sharpe = summary["Sharpe Ratio"].idxmax()
    best_ret    = summary["Ann. Return %"].idxmax()
    lowest_dd   = summary["Max Drawdown %"].idxmax()
    print(f"  • Best risk-adjusted return (Sharpe): {best_sharpe}  ({summary.loc[best_sharpe,'Sharpe Ratio']:.2f})")
    print(f"  • Highest annual return             : {best_ret}    ({summary.loc[best_ret,'Ann. Return %']:.1f}%)")
    print(f"  • Smallest max drawdown             : {lowest_dd}   ({summary.loc[lowest_dd,'Max Drawdown %']:.1f}%)")
    print(f"  • Stocks are {returns.corr().values[np.triu_indices(5,1)].mean():.2f} corr on avg — moderate diversification benefit.")
    print(f"  • Ridge model predicts next-day returns with RMSE={rmse:.5f}  R²={r2:.4f}")
    print(f"  • Low R² is expected — markets are noisy; directional accuracy matters more.")


# ══════════════════════════════════════════════════════════════════════════════
# ██  DOMAIN 2 — HEALTH
# ══════════════════════════════════════════════════════════════════════════════

def run_health(out):
    section("HEALTH  ·  Patient Readmission Prediction")

    df = make_health_data()
    print(f"\n  Dataset: {df.shape[0]} patients × {df.shape[1]} features")
    print(f"  Readmission rate: {df['readmitted'].mean()*100:.1f}%")
    print("\n  Statistical Summary (numeric):")
    print(df.describe().round(2).to_string())

    # ── Encode categoricals ───────────────────────────────────────────────────
    le_gender = LabelEncoder(); le_diag = LabelEncoder()
    df["gender_enc"]    = le_gender.fit_transform(df["gender"])
    df["diagnosis_enc"] = le_diag.fit_transform(df["diagnosis"])

    feat_cols = ["age","bmi","blood_pressure","glucose","hba1c",
                 "num_medications","prev_admissions","length_of_stay",
                 "gender_enc","diagnosis_enc"]
    X = df[feat_cols]; y = df["readmitted"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)

    models = {
        "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                          ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        "Decision Tree":       Pipeline([("sc", StandardScaler()),
                                          ("clf", DecisionTreeClassifier(max_depth=6, random_state=42))]),
        "Random Forest":       Pipeline([("sc", StandardScaler()),
                                          ("clf", RandomForestClassifier(n_estimators=120, random_state=42))]),
        "Gradient Boosting":   Pipeline([("sc", StandardScaler()),
                                          ("clf", GradientBoostingClassifier(n_estimators=120, random_state=42))]),
    }

    results = {}
    print("\n  Model Evaluation:")
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        ypred = mdl.predict(Xte)
        acc   = accuracy_score(yte, ypred)
        cv    = cross_val_score(mdl, X, y, cv=5, scoring="accuracy").mean()
        results[name] = {"acc": acc, "cv": cv, "pred": ypred, "model": mdl}
        print(f"  {name:<25}  Acc={acc:.4f}  CV-Acc={cv:.4f}")

    best_name = max(results, key=lambda n: results[n]["acc"])
    best_mdl  = results[best_name]["model"]
    print(f"\n  Best model: {best_name}")
    print(classification_report(yte, results[best_name]["pred"],
                                 target_names=["Not Readmitted", "Readmitted"]))

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)
    fig.suptitle("Health Dashboard — Patient Readmission Analysis",
                 fontsize=16, fontweight="bold", color=C["dark"], y=0.98)

    # 1. Readmission by diagnosis
    ax = fig.add_subplot(gs[0, 0])
    grp = df.groupby("diagnosis")["readmitted"].mean().sort_values(ascending=False) * 100
    ax.bar(grp.index, grp.values, color=PAL[:len(grp)], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_xticklabels(grp.index, rotation=20, ha="right")
    _title(ax, "Readmission Rate by Diagnosis")

    # 2. Age distribution by readmission
    ax2 = fig.add_subplot(gs[0, 1])
    for val, label, color in [(0,"Not Readmitted",C["teal"]),(1,"Readmitted",C["red"])]:
        ax2.hist(df.loc[df["readmitted"]==val,"age"], bins=25,
                 alpha=0.65, color=color, label=label, edgecolor="white")
    ax2.set_xlabel("Age"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    _title(ax2, "Age Distribution by Readmission Status")

    # 3. Feature importance
    ax3 = fig.add_subplot(gs[1, 0])
    clf = best_mdl.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values()
        colors_imp = [C["red"] if imp[f] == imp.max() else C["blue"] for f in imp.index]
        ax3.barh(imp.index, imp.values, color=colors_imp, edgecolor="white")
    _title(ax3, f"Feature Importance — {best_name}")
    ax3.set_xlabel("Importance Score")

    # 4. Confusion matrix
    ax4 = fig.add_subplot(gs[1, 1])
    cm  = confusion_matrix(yte, results[best_name]["pred"])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax4,
                cmap=sns.light_palette(C["blue"], as_cmap=True),
                xticklabels=["Not Read.","Readmitted"],
                yticklabels=["Not Read.","Readmitted"],
                cbar=False, annot_kws={"size": 13, "weight": "bold"})
    ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
    _title(ax4, f"Confusion Matrix — {best_name}")

    # 5. ROC curves
    ax5 = fig.add_subplot(gs[2, 0])
    for i, (name, res) in enumerate(results.items()):
        mdl_obj = res["model"]
        if hasattr(mdl_obj, "predict_proba"):
            yp = mdl_obj.predict_proba(Xte)[:, 1]
        else:
            yp = mdl_obj.decision_function(Xte)
        fpr, tpr, _ = roc_curve(yte, yp)
        roc_auc = auc(fpr, tpr)
        ax5.plot(fpr, tpr, color=PAL[i], lw=2,
                 label=f"{name} (AUC={roc_auc:.3f})")
    ax5.plot([0,1],[0,1], "k--", lw=1)
    ax5.set_xlabel("False Positive Rate"); ax5.set_ylabel("True Positive Rate")
    ax5.legend(fontsize=8, loc="lower right")
    _title(ax5, "ROC Curves — All Models")

    # 6. Model accuracy comparison
    ax6 = fig.add_subplot(gs[2, 1])
    names = list(results.keys())
    acc_v = [results[n]["acc"] for n in names]
    cv_v  = [results[n]["cv"]  for n in names]
    x = np.arange(len(names)); w = 0.38
    ax6.bar(x - w/2, acc_v, w, color=C["blue"],  label="Test Acc",  alpha=0.88)
    ax6.bar(x + w/2, cv_v,  w, color=C["teal"],  label="CV-5 Acc", alpha=0.88)
    ax6.set_xticks(x)
    ax6.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=8)
    ax6.set_ylim(0.5, 1.0)
    ax6.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax6.legend(fontsize=8)
    _title(ax6, "Model Accuracy Comparison")

    # 7. BMI vs Glucose scatter
    ax7 = fig.add_subplot(gs[3, 0])
    for val, label, col in [(0,"Not Readmitted",C["teal"]),(1,"Readmitted",C["red"])]:
        m = df["readmitted"] == val
        ax7.scatter(df.loc[m,"bmi"], df.loc[m,"glucose"],
                    alpha=0.35, s=18, color=col, label=label)
    ax7.set_xlabel("BMI"); ax7.set_ylabel("Glucose")
    ax7.legend(fontsize=8)
    _title(ax7, "BMI vs Glucose — Readmission Status")

    # 8. Prev admissions vs readmission rate
    ax8 = fig.add_subplot(gs[3, 1])
    grp2 = df.groupby("prev_admissions")["readmitted"].mean() * 100
    ax8.bar(grp2.index, grp2.values, color=C["purple"], edgecolor="white")
    ax8.set_xlabel("Previous Admissions"); ax8.set_ylabel("Readmission Rate (%)")
    _title(ax8, "Readmission Rate by Prior Admissions")

    _save(fig, os.path.join(out, "health_dashboard.png"))

    section("HEALTH — KEY CONCLUSIONS")
    best_acc = results[best_name]["acc"]
    top_diag = grp.idxmax()
    print(f"  • Dataset: {df.shape[0]} patients, {df['readmitted'].mean()*100:.1f}% readmission rate.")
    print(f"  • Highest-risk diagnosis     : {top_diag} ({grp[top_diag]:.1f}% readmission rate)")
    print(f"  • Prev. admissions strongly predicts readmission — rises steadily with count.")
    print(f"  • Best predictive model      : {best_name}  (Acc={best_acc:.4f})")
    if hasattr(clf, "feature_importances_"):
        top_feat = pd.Series(clf.feature_importances_, index=feat_cols).idxmax()
        print(f"  • Most influential feature   : {top_feat}")
    print(f"  • Recommendations: target high-risk patients (elderly, high glucose, repeat admissions)")
    print(f"    for early intervention and discharge-planning programmes.")


# ══════════════════════════════════════════════════════════════════════════════
# ██  DOMAIN 3 — RETAIL
# ══════════════════════════════════════════════════════════════════════════════

def run_retail(out):
    section("RETAIL  ·  Customer Lifetime Value & Segmentation")

    df = make_retail_data()
    print(f"\n  Dataset: {df.shape[0]} customers × {df.shape[1]} features")
    print(f"\n  CLV Summary:")
    print(df["clv"].describe().round(2).to_string())
    print(f"\n  Segment distribution:")
    print(df["segment"].value_counts().to_string())

    # ── CLV Prediction (regression) ───────────────────────────────────────────
    le_cat = LabelEncoder(); le_reg = LabelEncoder(); le_ch = LabelEncoder()
    df["cat_enc"] = le_cat.fit_transform(df["category"])
    df["reg_enc"] = le_reg.fit_transform(df["region"])
    df["ch_enc"]  = le_ch.fit_transform(df["channel"])

    feat_cols = ["tenure_months","num_orders","avg_order_value",
                 "return_rate","email_open_rate","discount_usage",
                 "cat_enc","reg_enc","ch_enc"]
    X = df[feat_cols]; y = df["clv"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_models = {
        "Linear Regression":    Pipeline([("sc", StandardScaler()),
                                            ("reg", LinearRegression())]),
        "Random Forest":        Pipeline([("sc", StandardScaler()),
                                            ("reg", RandomForestRegressor(n_estimators=120, random_state=42))]),
        "Gradient Boosting":    Pipeline([("sc", StandardScaler()),
                                            ("reg", GradientBoostingRegressor(n_estimators=120, random_state=42))]),
    }

    reg_results = {}
    print("\n  Regression Models:")
    for name, mdl in reg_models.items():
        mdl.fit(Xtr, ytr)
        ypred = mdl.predict(Xte)
        mae   = mean_absolute_error(yte, ypred)
        rmse  = np.sqrt(mean_squared_error(yte, ypred))
        r2    = r2_score(yte, ypred)
        reg_results[name] = {"mae": mae, "rmse": rmse, "r2": r2,
                              "pred": ypred, "model": mdl}
        print(f"  {name:<25}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    best_reg = max(reg_results, key=lambda n: reg_results[n]["r2"])
    print(f"  Best model: {best_reg}")

    # ── K-Means Customer Segmentation ─────────────────────────────────────────
    seg_feats = ["tenure_months","num_orders","avg_order_value",
                 "return_rate","email_open_rate"]
    scaler    = StandardScaler()
    Xseg      = scaler.fit_transform(df[seg_feats])

    inertias = []
    K_range  = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xseg)
        inertias.append(km.inertia_)

    km_best = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["km_cluster"] = km_best.fit_predict(Xseg)

    pca  = PCA(n_components=2, random_state=42)
    Xpca = pca.fit_transform(Xseg)

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 26))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.50, wspace=0.33)
    fig.suptitle("Retail Dashboard — Customer Lifetime Value & Segmentation",
                 fontsize=16, fontweight="bold", color=C["dark"], y=0.98)

    # 1. CLV distribution by segment
    ax = fig.add_subplot(gs[0, 0])
    seg_order = ["Bronze","Silver","Gold","Platinum"]
    seg_clrs  = [C["amber"], C["gray"], C["teal"], C["purple"]]
    for seg, col in zip(seg_order, seg_clrs):
        vals = df.loc[df["segment"]==seg,"clv"]
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.7, color=col, label=seg, edgecolor="white")
    ax.set_xlabel("CLV ($)"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    _title(ax, "CLV Distribution by Segment")

    # 2. Revenue by category & channel
    ax2 = fig.add_subplot(gs[0, 1])
    grp = df.groupby(["category","channel"])["clv"].mean().unstack()
    grp.plot(kind="bar", ax=ax2, color=PAL[:3], edgecolor="white", width=0.7)
    ax2.set_xlabel(""); ax2.set_ylabel("Avg CLV ($)")
    ax2.set_xticklabels(grp.index, rotation=20, ha="right")
    ax2.legend(fontsize=8)
    _title(ax2, "Avg CLV by Category & Channel")

    # 3. Elbow plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(list(K_range), inertias, "o-", color=C["blue"], lw=2, markersize=7)
    ax3.axvline(4, color=C["red"], ls="--", lw=1.5, label="Chosen k=4")
    ax3.set_xlabel("Number of Clusters k"); ax3.set_ylabel("Inertia")
    ax3.legend(fontsize=8)
    _title(ax3, "Elbow Method — Optimal k")

    # 4. PCA cluster scatter
    ax4 = fig.add_subplot(gs[1, 1])
    for k, col in zip(range(4), PAL):
        m = df["km_cluster"] == k
        ax4.scatter(Xpca[m, 0], Xpca[m, 1], c=col, s=18,
                    alpha=0.6, label=f"Cluster {k}")
    ax4.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax4.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax4.legend(fontsize=8)
    _title(ax4, "Customer Segments (PCA 2-D)")

    # 5. Cluster profile heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    profile = df.groupby("km_cluster")[seg_feats].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
    sns.heatmap(profile_norm.T, annot=profile.T.round(1), fmt=".1f", ax=ax5,
                cmap="Blues", linewidths=0.4, cbar=False, annot_kws={"size": 8})
    ax5.set_xlabel("Cluster"); ax5.set_title("")
    _title(ax5, "Cluster Feature Profiles (Normalised)")

    # 6. Feature importance (best regression model)
    ax6 = fig.add_subplot(gs[2, 1])
    reg_clf = reg_results[best_reg]["model"].named_steps["reg"]
    if hasattr(reg_clf, "feature_importances_"):
        imp = pd.Series(reg_clf.feature_importances_, index=feat_cols).sort_values()
        ax6.barh(imp.index, imp.values,
                 color=[C["red"] if v == imp.max() else C["blue"] for v in imp.values],
                 edgecolor="white")
        ax6.set_xlabel("Importance Score")
    _title(ax6, f"Feature Importance — {best_reg}")

    # 7. Predicted vs Actual CLV
    ax7 = fig.add_subplot(gs[3, 0])
    ypred_best = reg_results[best_reg]["pred"]
    ax7.scatter(yte, ypred_best, alpha=0.35, s=16, color=C["blue"])
    mn, mx = yte.min(), yte.max()
    ax7.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    ax7.set_xlabel("Actual CLV ($)"); ax7.set_ylabel("Predicted CLV ($)")
    ax7.legend(fontsize=8)
    r2 = reg_results[best_reg]["r2"]
    _title(ax7, f"Predicted vs Actual CLV — {best_reg}  (R²={r2:.3f})")

    # 8. Segment revenue contribution
    ax8 = fig.add_subplot(gs[3, 1])
    seg_rev = df.groupby("segment")["clv"].sum().reindex(seg_order)
    wedge_colors = seg_clrs
    wedges, texts, autotexts = ax8.pie(
        seg_rev.values, labels=seg_order, colors=wedge_colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9}
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_fontweight("bold")
    _title(ax8, "Revenue Contribution by Segment")

    # 9. Tenure vs CLV
    ax9 = fig.add_subplot(gs[4, :])
    for seg, col in zip(seg_order, seg_clrs):
        m = df["segment"] == seg
        ax9.scatter(df.loc[m,"tenure_months"], df.loc[m,"clv"],
                    alpha=0.30, s=16, color=col, label=seg)
    # regression line
    slope, intercept, *_ = stats.linregress(df["tenure_months"], df["clv"])
    xline = np.linspace(1, 72, 200)
    ax9.plot(xline, slope * xline + intercept, color=C["dark"],
             lw=2, ls="--", label=f"Trend (r²={stats.pearsonr(df['tenure_months'],df['clv'])[0]**2:.2f})")
    ax9.set_xlabel("Tenure (months)"); ax9.set_ylabel("CLV ($)")
    ax9.legend(fontsize=8, ncol=5)
    _title(ax9, "Customer Tenure vs Lifetime Value")

    _save(fig, os.path.join(out, "retail_dashboard.png"))

    section("RETAIL — KEY CONCLUSIONS")
    best_seg = df.groupby("segment")["clv"].mean().idxmax()
    best_cat = df.groupby("category")["clv"].mean().idxmax()
    best_ch  = df.groupby("channel")["clv"].mean().idxmax()
    r2_val   = reg_results[best_reg]["r2"]
    print(f"  • Dataset: {df.shape[0]} customers, avg CLV = ${df['clv'].mean():.2f}")
    print(f"  • Highest-value segment : {best_seg}")
    print(f"  • Best-performing category & channel: {best_cat} via {best_ch}")
    print(f"  • K-Means (k=4) reveals 4 distinct customer personas.")
    print(f"  • Best CLV prediction model: {best_reg}  (R²={r2_val:.4f})")
    if hasattr(reg_clf, "feature_importances_"):
        top_feat = pd.Series(reg_clf.feature_importances_, index=feat_cols).idxmax()
        print(f"  • Strongest CLV driver: {top_feat}")
    print(f"  • Recommendations:")
    print(f"      – Invest in retention programmes for Bronze/Silver customers.")
    print(f"      – Reward Platinum customers to reduce churn risk.")
    print(f"      – Increase email engagement — strong positive CLV correlation.")
    print(f"      – Reduce return rates; each % point reduces CLV significantly.")


# ══════════════════════════════════════════════════════════════════════════════
# ██  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

RUNNERS = {"finance": run_finance, "health": run_health, "retail": run_retail}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-World Data Science Project")
    parser.add_argument("--domain",
                        choices=["finance", "health", "retail", "all"],
                        default="all",
                        help="Domain to run (default: all)")
    parser.add_argument("--output", default="realworld_output",
                        help="Output directory for plots (default: realworld_output)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.domain == "all":
        for name, fn in RUNNERS.items():
            fn(args.output)
    else:
        RUNNERS[args.domain](args.output)

    print(f"\n{'═'*65}")
    print(f"  All outputs saved to: {args.output}/")
    print(f"{'═'*65}\n")
