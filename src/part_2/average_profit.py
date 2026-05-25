import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "data/sales.csv",
    parse_dates=["Date"],
)
df["GrossProfit"] = df["Revenue"] - df["COGS"]
df["Month"] = df["Date"].dt.month

monthly_avg = (
    df.groupby("Month")["GrossProfit"]
    .mean()
    .reset_index()
)

# ── Style constants ───────────────────────────────────────────────────────────
MONTH_LABELS = [f"T{i}" for i in range(1, 13)]
PEAK_MONTHS  = {4, 5, 6}          # tháng cao điểm, highlight đậm hơn
COLOR_PEAK   = "#2D7D46"           # xanh lá đậm
COLOR_REST   = "#A8D5B5"           # xanh lá nhạt
ACCENT_LINE  = "#1A4D2E"           # màu đường + điểm
GRID_COLOR   = "#E8E8E8"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x      = monthly_avg["Month"].values          # 1..12
values = monthly_avg["GrossProfit"].values

bar_colors = [COLOR_PEAK if m in PEAK_MONTHS else COLOR_REST for m in x]

bars = ax.bar(
    x, values,
    color=bar_colors,
    width=0.62,
    zorder=2,
    linewidth=0,
)

# Data labels trên mỗi cột
for xi, vi in zip(x, values):
    label = f"{vi/1e6:.1f}M"
    ax.text(xi, vi + max(values) * 0.018, label,
            ha="center", va="bottom",
            fontsize=8.5, color="#333333", fontweight="500")

# ── Axes & grid ───────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS, fontsize=9)
y_max = max(values)
tick_step = 0.2e6 if y_max <= 1.4e6 else 0.5e6
ticks = np.arange(0, y_max * 1.21, tick_step)

ax.set_yticks(ticks)

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(
        lambda val, _: f"{val/1e6:.1f}M" if val > 0 else "0M"
    )
)
ax.tick_params(axis="y", labelsize=9, colors="#555555")
ax.tick_params(axis="x", colors="#333333", length=0)

ax.set_ylim(0, ticks[-1])
ax.set_xlabel("Tháng", fontsize=10, color="#444444", labelpad=8, fontweight=500)
ax.set_ylabel("Lợi nhuận gộp TB", fontsize=10,
              color="#444444", labelpad=10, fontweight=500)

ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(GRID_COLOR)

# ── Peak highlight band ───────────────────────────────────────────────────────
ax.axvspan(3.62, 6.38, color="#F0FAF3", zorder=0, label="Giai đoạn cao điểm (T4–T6)")

# ── Legend ────────────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_PEAK,  label="Tháng cao điểm (T4–T6)"),
    Patch(facecolor=COLOR_REST,  label="Các tháng còn lại"),
]
ax.legend(handles=legend_elements, loc="upper right",
          frameon=True, framealpha=0.9, edgecolor=GRID_COLOR,
          fontsize=9, handlelength=1.2, handleheight=0.9)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Lợi nhuận gộp trung bình theo Tháng (2012–2022)",
    fontsize=13, fontweight="bold", color="#1A1A1A", pad=14,
)

plt.tight_layout()
plt.savefig(
    "reports/figures/monthly_avg_profit.png",
    dpi=180, bbox_inches="tight", facecolor="white",
)
plt.show()
print("Saved → monthly_avg_profit.png")
