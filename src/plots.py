"""Required research figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Headless-safe: figures are saved to disk, never opened interactively.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def actual_vs_predicted(predictions: pd.DataFrame, output, max_buildings=3):
    subset = predictions[predictions["experiment"] == "unseen"].copy()
    ids = subset["building_id"].drop_duplicates().head(max_buildings)
    models = [m for m in ("persistence", "linear_regression", "lightgbm", "xgboost") if m in subset["model"].unique()]
    fig, axes = plt.subplots(len(ids), 1, figsize=(13, 3.5 * len(ids)), squeeze=False)
    for ax, building in zip(axes[:, 0], ids):
        group = subset[subset["building_id"] == building]
        end = group["timestamp"].max()
        grid = pd.date_range(end=end, periods=168, freq="h")
        actual = group.drop_duplicates("timestamp").set_index("timestamp")["target"].reindex(grid)
        ax.plot(grid, actual, label="actual", color="black", linewidth=1.5)
        for model in models:
            model_data = group[group["model"] == model].set_index("timestamp")["prediction"].reindex(grid)
            ax.plot(grid, model_data, label=model, alpha=.8)
        ax.set_title(str(building)); ax.set_ylabel("Electricity")
    axes[0, 0].legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def model_comparison(summary: pd.DataFrame, output):
    data = summary[summary["experiment"] == "unseen"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="model", y="macro_MAE", ax=ax)
    ax.set(title="Unseen-building model comparison", xlabel="", ylabel="Macro building MAE")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def seen_vs_unseen(summary: pd.DataFrame, output):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=summary, x="model", y="macro_MAE", hue="experiment", ax=ax)
    ax.set(title="Seen vs unseen building performance", xlabel="", ylabel="Macro building MAE")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def error_distribution(per_building: pd.DataFrame, output):
    data = per_building[per_building["experiment"] == "unseen"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=data, x="model", y="MAE", ax=ax, showfliers=False)
    ax.set(title="Distribution of unseen-building errors", xlabel="", ylabel="Building MAE")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def importance_plot(importance: pd.DataFrame, output, top_n=20):
    data = importance.nlargest(top_n, "importance").sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(data["feature"], data["importance"])
    ax.set(title="Tree-model feature importance", xlabel="Feature importance")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)
