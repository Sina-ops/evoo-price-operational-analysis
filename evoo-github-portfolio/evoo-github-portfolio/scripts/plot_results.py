from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# These CSVs contain reported thesis metrics, not raw monthly observations.
holdout = pd.read_csv(DATA / "holdout_metrics.csv").sort_values("MAE_EUR_per_100kg")
expanding = pd.read_csv(DATA / "expanding_window_metrics.csv").sort_values("MAE_EUR_per_100kg")
regime = pd.read_csv(DATA / "regime_mae.csv")
shap = pd.read_csv(DATA / "xgboost_c_shap.csv").sort_values("mean_absolute_SHAP")

def bar_chart(df, x, y, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df[x], df[y])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)

bar_chart(holdout, "model", "MAE_EUR_per_100kg",
          "Recent chronological holdout: MAE by model",
          "MAE (€ per 100 kg)", "holdout_mae.png")
bar_chart(expanding, "model", "MAE_EUR_per_100kg",
          "Expanding-window backtest: MAE by model",
          "MAE (€ per 100 kg)", "expanding_window_mae.png")

fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(regime))
width = 0.25
for i, col in enumerate(["Low_MAE", "Medium_MAE", "High_MAE"]):
    ax.bar([v + (i - 1) * width for v in x], regime[col], width, label=col.replace("_MAE", ""))
ax.set_xticks(list(x), regime["model"], rotation=25, ha="right")
ax.set_ylabel("MAE (€ per 100 kg)")
ax.set_title("Expanding-window MAE by price regime")
ax.legend(title="Price regime")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "mae_by_price_regime.png", dpi=180)
plt.close(fig)

# Current EVOO price is shown here because this chart represents the reported global SHAP ranking.
bar_chart(shap, "feature", "mean_absolute_SHAP",
          "XGBoost C: reported mean absolute SHAP values",
          "Mean absolute SHAP value", "xgboost_c_shap.png")
print(f"Charts saved to: {OUT}")
