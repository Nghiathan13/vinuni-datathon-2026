import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "data/sales.csv",
    parse_dates=["Date"],
)
df["Year"] = df["Date"].dt.year

annual = (
    df[df["Year"].between(2013, 2022)]
    .groupby("Year")["Revenue"]
    .sum()
    .reset_index()
)

# ── Style ─────────────────────────────────────────────────────────────────────
BAR_COLOR  = "#2D7D46"     # dùng 1 màu duy nhất
GRID_COLOR = "#EBEBEB"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

years  = annual["Year"].values
values = annual["Revenue"].values

# width tăng nhẹ cho cột đầy chart hơn
bars = ax.bar(
    years,
    values,
    color=BAR_COLOR,
    width=0.72,
    zorder=2,
    linewidth=0
)

# Data labels (2 số lẻ)
for yr, val in zip(years, values):
    ax.text(
        yr,
        val + max(values) * 0.015,
        f"{val/1e9:.2f}B",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#222222",
        fontweight="500"
    )

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)

ax.set_xticks(years)
ax.set_xticklabels(years, fontsize=9, color="#333333")
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelsize=9, colors="#555555")

# Trục Y: tick cố định và luôn dư 1 mốc phía trên
y_max = max(values)

tick_step = 0.25e9 if y_max <= 1.5e9 else 0.5e9

upper_tick = np.ceil(y_max / tick_step) * tick_step

ticks = np.arange(0, upper_tick + tick_step/2, tick_step)

ax.set_yticks(ticks)

ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(
        lambda v, _: f"{v/1e9:.1f}B" if v > 0 else "0B"
    )
)

ax.set_ylim(0, ticks[-1])

ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(GRID_COLOR)

ax.set_xlabel("Năm", fontsize=10, color="#444444", labelpad=10)
ax.set_ylabel("Doanh thu", fontsize=10, color="#444444", labelpad=10)

# Title mới
ax.set_title(
    "Doanh thu giai đoạn 2013–2022",
    fontsize=13,
    fontweight="bold",
    color="#1A1A1A",
    pad=14,
)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(
    "reports/figures/annual_revenue.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

print("Saved → annual_revenue.png")
