import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── Shared style ──────────────────────────────────────────────────────────────
COLOR_EVEN   = "#2D7D46"   # năm chẵn / tháng cao điểm
COLOR_ODD    = "#A8D5B5"   # năm lẻ  / tháng thường
COLOR_LOW    = "#E8B84B"   # tháng thấp điểm (T8, T12)
GRID_COLOR   = "#EBEBEB"
LABEL_COLOR  = "#222222"
AXIS_COLOR   = "#444444"
TICK_COLOR   = "#555555"

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "data/sales.csv",
    parse_dates=["Date"],
)
df["Year"]        = df["Date"].dt.year
df["Month"]       = df["Date"].dt.month
df["GrossProfit"] = df["Revenue"] - df["COGS"]

# Lọc dữ liệu từ năm 2013 trở đi
df = df[df["Year"] >= 2013].copy()

# Chart trái: lợi nhuận gộp TB theo tháng (Tổng tháng trung bình qua các năm)
monthly_sum = df.groupby(["Year", "Month"])["GrossProfit"].sum().reset_index()
monthly_avg = (
    monthly_sum.groupby("Month")["GrossProfit"]
    .mean()
    .reset_index()
)

# Chart phải: tổng lợi nhuận gộp theo năm 2013–2022
annual = (
    df[df["Year"].between(2013, 2022)]
    .groupby("Year")["GrossProfit"]
    .sum()
    .reset_index()
)
annual["IsEven"] = annual["Year"] % 2 == 0

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))
fig.patch.set_facecolor("white")


# ── Helper: áp style chung cho 1 ax ──────────────────────────────────────────
def apply_common_style(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=9, colors=TICK_COLOR)


# ════════════════════════════════════════════════════════════════════════════
# Chart trái — Lợi nhuận gộp trung bình theo Tháng (2012–2022)
# ════════════════════════════════════════════════════════════════════════════
MONTH_LABELS = [f"T{i}" for i in range(1, 13)]
PEAK_MONTHS  = {4, 5, 6}
LOW_MONTHS   = {8, 12}

x      = monthly_avg["Month"].values
profit = monthly_avg["GrossProfit"].values

def month_color(m):
    if m in PEAK_MONTHS: return COLOR_EVEN
    if m in LOW_MONTHS:  return COLOR_LOW
    return COLOR_ODD

colors = [month_color(m) for m in x]

ax1.bar(x, profit, color=colors, width=0.62, zorder=2, linewidth=0)

# Peak highlight band
ax1.axvspan(3.62, 6.38, color="#F0FAF3", zorder=0)

# Data labels
for xi, vi in zip(x, profit):
    ax1.text(xi, vi + max(profit) * 0.018,
             f"{vi/1e6:.1f}M",
             ha="center", va="bottom",
             fontsize=8.5, color=LABEL_COLOR, fontweight="500")

# Trục Y: dynamic ticks (điều chỉnh cho quy mô triệu đơn vị)
y_max1     = max(profit)
tick_step1 = 5e6 if y_max1 <= 25e6 else 10e6
ticks1     = np.arange(0, y_max1 * 1.25, tick_step1)

ax1.set_yticks(ticks1)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v > 0 else "0M")
)
ax1.set_ylim(0, ticks1[-1])
ax1.set_xlim(x[0] - 0.6, x[-1] + 0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(MONTH_LABELS, fontsize=9, color="#333333")

apply_common_style(ax1)

# Đường đóng khung trái/phải
ax1.axvline(x[0]  - 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)
ax1.axvline(x[-1] + 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)

ax1.set_xlabel("Tháng", fontsize=10, color=AXIS_COLOR, labelpad=8, fontweight=500)
ax1.set_ylabel("Lợi nhuận gộp TB", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight=500)

legend_elements = [
    mpatches.Patch(facecolor=COLOR_EVEN, label="Tháng cao điểm (T4–T6)"),
    mpatches.Patch(facecolor=COLOR_ODD,  label="Các tháng còn lại"),
    mpatches.Patch(facecolor=COLOR_LOW,  label="Tháng thấp điểm (T8, T12)"),
]
ax1.legend(handles=legend_elements, loc="upper right",
           frameon=True, framealpha=0.9, edgecolor=GRID_COLOR,
           fontsize=9, handlelength=1.2, handleheight=0.9)


# ════════════════════════════════════════════════════════════════════════════
# Chart phải — Tổng lợi nhuận gộp theo Năm (2012–2022)
# ════════════════════════════════════════════════════════════════════════════
years  = annual["Year"].values
gp     = annual["GrossProfit"].values
colors_bar = [COLOR_EVEN if e else COLOR_ODD for e in annual["IsEven"]]

ax2.bar(years, gp, color=colors_bar, width=0.72, zorder=2, linewidth=0)

# Data labels
for yr, val in zip(years, gp):
    ax2.text(yr, val + max(gp) * 0.015,
             f"{val/1e9:.2f}B",
             ha="center", va="bottom",
             fontsize=8.5, color=LABEL_COLOR, fontweight="500")

# Trục Y: tick cố định, luôn dư 1 mốc phía trên
y_max2     = max(gp)
tick_step2 = 0.05e9 if y_max2 <= 0.5e9 else 0.1e9
upper2     = np.ceil(y_max2 / tick_step2) * tick_step2
ticks2     = np.arange(0, upper2 + tick_step2 / 2, tick_step2)

ax2.set_yticks(ticks2)
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v/1e9:.2f}B" if v > 0 else "0B")
)
ax2.set_ylim(0, ticks2[-1])
ax2.set_xlim(years[0] - 0.6, years[-1] + 0.6)
ax2.set_xticks(years)
ax2.set_xticklabels(years, fontsize=9, color="#333333")

apply_common_style(ax2)

# Đường đóng khung trái/phải
ax2.axvline(years[0]  - 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)
ax2.axvline(years[-1] + 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)

ax2.set_xlabel("Năm", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight=500)
ax2.set_ylabel("Lợi nhuận gộp", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight=500)

legend_elements2 = [
    mpatches.Patch(facecolor=COLOR_EVEN, label="Năm chẵn"),
    mpatches.Patch(facecolor=COLOR_ODD,  label="Năm lẻ"),
]
ax2.legend(handles=legend_elements2, loc="upper right",
           frameon=True, framealpha=0.9, edgecolor=GRID_COLOR,
           fontsize=9, handlelength=1.2, handleheight=0.9)


# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(w_pad=4)
plt.savefig(
    "reports/figures/combined_charts.png",
    dpi=180, bbox_inches="tight", facecolor="white",
)
plt.show()
print("Saved → combined_charts.png")
