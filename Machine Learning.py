import json
import warnings
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.cm import ScalarMappable
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ------------------------- 修改这里 -------------------------
DATA_FILE = Path("chili_oil_features.csv")  # CSV 或 Excel 的实际路径
TARGET_COLUMN = "grade"
SHEET_NAME = 0
ID_COLUMNS = ["user_id", "date", "Sample_ID", "sample_id", "Batch_ID", "batch_id"]
LABEL_CODES = None  # 标签为文字时，填写原分析的编码字典；不要随意改变编码
STANDARDIZE = True  # 论文的标准化处理；已处理的输入须核对是否还需要此步骤

SVR_PARAMS = None  # 有原来的最终 C、gamma、epsilon 时，在此填写参数字典
N_TRIALS = 100
CV_FOLDS = 3
TEST_SIZE = 0.3  # 沿用原脚本；n=9 时为 6 个训练批次、3 个作图批次
RANDOM_SEED = 0

TOP_FEATURES = 15
DPI = 600
OUTPUT_DIR = Path("SVR_SHAP_results")
# ------------------------------------------------------------


def load_data():
    if DATA_FILE.suffix.lower() == ".csv":
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    if TARGET_COLUMN not in df:
        raise ValueError(f"目标列 {TARGET_COLUMN!r} 不存在。")
    y = df[TARGET_COLUMN]
    if LABEL_CODES is not None:
        y = y.map(LABEL_CODES)
    y = pd.to_numeric(y, errors="raise")
    X = df.drop(columns=[TARGET_COLUMN, *ID_COLUMNS], errors="ignore")
    X = X.apply(pd.to_numeric, errors="raise").astype(float)
    if X.shape[1] == 0 or not np.isfinite(X.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError("预测变量不能为空；输入必须为无缺失值的有限数值。")
    counts = y.value_counts()
    if len(X) != 9 or len(counts) != 3 or not counts.eq(3).all():
        raise ValueError("论文的数据结构应为 9 个批次、3 个数值标签、每组 3 个批次。")
    return X, y


def make_model(params):
    svr = SVR(**params)
    return make_pipeline(StandardScaler(), svr) if STANDARDIZE else svr


def fit_svr(X, y):
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(splitter.split(X, y))
    search_space = {"C": [0.1, 100.0], "C_log": True,
                    "gamma": ["scale", "auto"], "epsilon": [0.01, 1.0]}
    tuning = {"sampler": "TPESampler", "seed": RANDOM_SEED,
              "search_space": search_space, "scope": "all nine batches; internal only"}
    if SVR_PARAMS is None:
        def objective(trial):
            params = {"kernel": "rbf",
                      "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                      "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                      "epsilon": trial.suggest_float("epsilon", 0.01, 1.0)}
            return float(cross_val_score(make_model(params), X, y, cv=folds,
                                         scoring="r2", error_score="raise").mean())
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=N_TRIALS)
        params = {"kernel": "rbf", **study.best_params}
        tuning.update(trials=len(study.trials), best_internal_cv_r2=study.best_value)
    else:
        if not {"C", "gamma", "epsilon"}.issubset(SVR_PARAMS):
            raise ValueError("SVR_PARAMS 必须包含原分析的 C、gamma、epsilon。")
        params = {"kernel": "rbf", **SVR_PARAMS}
        if params["kernel"] != "rbf":
            raise ValueError("论文指定的是 RBF 核。")
        tuning = {"trials": 0, "source": "user-supplied historical parameters"}
    train_rows, plot_rows = train_test_split(np.arange(len(X)), test_size=TEST_SIZE,
                                           random_state=RANDOM_SEED)
    model = make_model(params).fit(X.iloc[train_rows], y.iloc[train_rows])
    return model, params, tuning, folds, train_rows, plot_rows


def evaluate_cv(X, y, params, folds):
    results = cross_validate(make_model(params), X, y, cv=folds,
                             scoring={"R2": "r2", "RMSE": "neg_root_mean_squared_error",
                                      "MAE": "neg_mean_absolute_error"}, error_score="raise")
    fold_scores = pd.DataFrame({"Fold": np.arange(1, len(folds) + 1),
                                "R2": results["test_R2"],
                                "RMSE": -results["test_RMSE"],
                                "MAE": -results["test_MAE"]})
    summary = {metric: {"mean": float(fold_scores[metric].mean()),
                        "sd": float(fold_scores[metric].std(ddof=0))}
               for metric in ["R2", "RMSE", "MAE"]}
    table = pd.concat([fold_scores, pd.DataFrame([
        {"Fold": "Mean", **{metric: stats["mean"] for metric, stats in summary.items()}},
        {"Fold": "SD", **{metric: stats["sd"] for metric, stats in summary.items()}}
    ])], ignore_index=True)
    report = {"n_folds": len(folds), "standard_deviation_ddof": 0,
              "fold_scores": fold_scores.to_dict(orient="records"), "summary": summary,
              "scope": "descriptive internal assessment of the selected SVR parameters"}
    return table, report


def plot_combined(explanation, X_plot, output_file):
    values = np.asarray(explanation.values)
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(-importance, kind="stable")
    keep, remaining = order[:TOP_FEATURES], order[TOP_FEATURES:]
    names = X_plot.columns[keep].tolist()
    shown_values = values[:, keep]
    shown_data = X_plot.iloc[:, keep].to_numpy()
    if len(remaining):
        # Other 的条形和散点都来自同一个有符号 SHAP 合计，保持两图一致。
        shown_values = np.column_stack([shown_values, values[:, remaining].sum(axis=1)])
        shown_data = np.column_stack([shown_data, np.full(len(X_plot), np.nan)])
        names.append("Other")
    means = np.abs(shown_values).mean(axis=0)
    if means.sum() == 0:
        raise ValueError("所有 SHAP 值均为零，无法计算玫瑰图的比例。")
    n = len(names)
    ypos = np.arange(n)[::-1]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.unicode_minus": False})
    base_cmap = plt.get_cmap("coolwarm")
    bar_cmap = colors.LinearSegmentedColormap.from_list(
        "coolwarm_trimmed", base_cmap(np.linspace(0.2, 0.9, 256)))
    low, high = np.quantile(means, [0.25, 0.75])
    if low == high:
        low, high = 0.0, float(means.max())
    norm = colors.Normalize(low, high, clip=True)
    bar_colors = bar_cmap(norm(means))

    fig = plt.figure(figsize=(20, max(9, 0.75 * n + 1.5)), facecolor="white")
    bottom, height = 0.13, 0.81
    ax_bar = fig.add_axes([0.13, bottom, 0.39, height])
    ax_names = fig.add_axes([0.52, bottom, 0.20, height])
    ax_bee = fig.add_axes([0.72, bottom, 0.22, height])
    ax_bar.barh(ypos, means, color=bar_colors, height=0.64, edgecolor="white")
    ax_bar.set_xlim(means.max() * 1.32, 0)
    ax_bar.set_ylim(-0.75, n - 0.25)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Mean|SHAP Value|", fontsize=20, labelpad=15)
    ax_bar.spines[["left", "top"]].set_visible(False)
    ax_bar.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax_bar.tick_params(axis="x", labelsize=16, direction="in", length=8, width=2)
    ax_bar.tick_params(axis="x", which="minor", direction="in", length=4)
    for spine in ax_bar.spines.values():
        spine.set_linewidth(2.5)
    ax_names.set_xlim(0, 1)
    ax_names.set_ylim(ax_bar.get_ylim())
    ax_names.axis("off")
    for pos, name, value in zip(ypos, names, means):
        ax_names.text(0.5, pos, name, ha="center", va="center", fontsize=13)
        ax_bar.text(value + means.max() * 0.025, pos, f"{value:.4f}",
                    ha="right", va="center", fontsize=14)

    # 左侧重要性色标。
    cax_left = fig.add_axes([0.075, bottom, 0.012, height])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=bar_cmap), cax=cax_left)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    for text, y, va in [("High", 1, "bottom"), ("Low", 0, "top")]:
        cax_left.text(-2.1, y, text, transform=cax_left.transAxes,
                      ha="center", va=va, fontsize=16)
    cax_left.text(-3.0, 0.5, "Mean|SHAP Value|", transform=cax_left.transAxes,
                  ha="center", va="center", rotation=90, fontsize=18)

    # 玫瑰图：角宽表示图中各行平均绝对 SHAP 值的比例。
    percentages = means / means.sum() * 100
    widths = percentages / 100 * 2 * np.pi
    angles = np.r_[0, np.cumsum(widths[:-1])] - np.pi / 21
    lengths = 4.0 + np.arange(n) * 0.5
    inner_heights = lengths - 2.0
    ax_rose = fig.add_axes([0.15, 0.24, 0.23, 0.41], projection="polar")
    ax_rose.patch.set_alpha(0)
    ax_rose.bar(angles, inner_heights, width=widths, align="edge",
                color=["#F5F5F5", "white"] * (n // 2) + (["#F5F5F5"] if n % 2 else []),
                edgecolor="white", linewidth=2)
    ax_rose.bar(angles, np.full(n, 2.0), width=widths, bottom=inner_heights,
                align="edge", color=bar_colors, edgecolor="white", linewidth=2)
    for i, (angle, width, radius, pct) in enumerate(zip(angles, widths, lengths, percentages)):
        if pct >= 2.0:
            ax_rose.text(angle + width / 2, radius + 0.6 + 0.8 * (i % 2), f"{pct:.2f}%",
                         ha="center", va="center", fontsize=7,
                         bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "pad": 1})
    ax_rose.set_xticks([])
    ax_rose.set_yticks([])
    ax_rose.spines["polar"].set_visible(False)
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)
    ax_rose.set_ylim(0, lengths.max() + 2.5)

    # 指定相同的行顺序，防止蜂群图重新排序后与左图错位。
    display = shap.Explanation(values=shown_values, base_values=explanation.base_values,
                               data=shown_data, feature_names=names)
    np.random.seed(RANDOM_SEED)
    with warnings.catch_warnings():
        # Other 没有单一特征值，NaN 用于画灰点；仅屏蔽对应的已知绘图提示。
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        shap.plots.beeswarm(display, max_display=n, order=np.arange(n), ax=ax_bee,
                            color=base_cmap, s=55, color_bar=False, plot_size=None, show=False)
    ax_bee.set_yticks([])
    ax_bee.set_ylabel("")
    ax_bee.set_ylim(ax_bar.get_ylim())
    ax_bee.set_xlabel("SHAP Value", fontsize=20, labelpad=15)
    ax_bee.tick_params(axis="x", labelsize=16, direction="in", length=8, width=2)
    ax_bee.spines["bottom"].set_linewidth(2.5)
    cax_right = fig.add_axes([0.955, bottom, 0.012, height])
    cbar_right = fig.colorbar(ScalarMappable(norm=colors.Normalize(0, 1), cmap=base_cmap),
                              cax=cax_right)
    cbar_right.set_ticks([0, 1], labels=["Low", "High"])
    cbar_right.set_label("Feature Value", rotation=270, labelpad=25, fontsize=18)
    cbar_right.ax.tick_params(labelsize=16, length=0)
    cbar_right.outline.set_visible(False)
    if len(remaining):
        fig.text(0.53, 0.035, "Other: sum of remaining signed SHAP values; grey points have no single feature value.",
                  ha="center", fontsize=11, fontweight="normal")
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"displayed_features": names, "other_members": X_plot.columns[remaining].tolist(),
            "other_aggregation": "per-row signed sum; importance = mean absolute sum"}


def main():
    X, y = load_data()
    model, params, tuning, folds, train_rows, plot_rows = fit_svr(X, y)
    cv_table, cv_report = evaluate_cv(X, y, params, folds)
    cv_report["tuned_on_same_folds_in_this_run"] = tuning["trials"] > 0
    background, X_plot = X.iloc[train_rows], X.iloc[plot_rows]
    # 沿用原组合图的 shap.Explainer 接口；只计算一次，补充随机种子。
    explainer = shap.Explainer(model.predict, background, seed=RANDOM_SEED)
    explanation = explainer(X_plot)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_file = OUTPUT_DIR / "shap_combined_coolwarm.jpg"
    display_info = plot_combined(explanation, X_plot, image_file)
    cv_table.to_csv(OUTPUT_DIR / "svr_cross_validation.csv", index=False, encoding="utf-8-sig")
    record = {"model": "SVR", "parameters": params, "standardize": STANDARDIZE,
              "input_file": str(DATA_FILE), "target_column": TARGET_COLUMN,
              "label_codes": LABEL_CODES, "numeric_labels_in_row_order": y.tolist(),
              "features": X.columns.tolist(), "tuning": tuning,
              "row_indexing": "zero-based data rows, excluding the header",
              "train_rows": train_rows.tolist(), "shap_rows": plot_rows.tolist(),
              "test_size": TEST_SIZE, "split_seed": RANDOM_SEED, "cv_seed": RANDOM_SEED,
              "cv_folds": [{"train": tr.tolist(), "validation": va.tolist()} for tr, va in folds],
              "cross_validation": cv_report,
              "shap": {"interface": "shap.Explainer", "algorithm": type(explainer).__name__,
                       "seed": RANDOM_SEED, "background_rows": train_rows.tolist(),
                       "max_evals": "SHAP default", **display_info},
              "plot": {"cmap": "coolwarm", "dpi": DPI, "top_features": TOP_FEATURES},
              "versions": {pkg: version(pkg) for pkg in
                           ["numpy", "pandas", "matplotlib", "scikit-learn", "shap", "optuna"]},
              "interpretation": "Exploratory within-sample ranking; no external validation."}
    (OUTPUT_DIR / "svr_shap_parameters.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"SVR 参数：{params}")
    print(f"分层 {CV_FOLDS} 折交叉验证（均值 ± 标准差）：")
    for metric, stats in cv_report["summary"].items():
        label = "R²" if metric == "R2" else metric
        print(f"{label}：{stats['mean']:.3f} ± {stats['sd']:.3f}")
    print(f"SHAP 图：{image_file.resolve()}")


if __name__ == "__main__":
    main()
