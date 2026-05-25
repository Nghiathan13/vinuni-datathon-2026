import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Shared style ──────────────────────────────────────────────────────────────
COLOR_BAR    = "#2D7D46"   # new customers — bar (xanh)
COLOR_LINE   = "#E8B84B"   # active customers — line (vàng)
COLOR_DOT    = "#C8960A"   # dot trên line (vàng đậm)
GRID_COLOR   = "#EBEBEB"
LABEL_COLOR  = "#222222"
AXIS_COLOR   = "#444444"
TICK_COLOR   = "#555555"

# ── 1. Data Processing ────────────────────────────────────────────────────────
print("Đang đọc dữ liệu và xử lý...")
orders = pd.read_csv(
    "data/orders.csv",
    parse_dates=["order_date"],
)
orders["Year"] = orders["order_date"].dt.year

# -- Data for Left Chart (Active & New Customers) --
first_yr = orders.groupby("customer_id")["Year"].min()

new_custs = (
    first_yr.reset_index()
    .rename(columns={"Year": "first_year"})
    .groupby("first_year")["customer_id"]
    .count()
    .reset_index()
    .rename(columns={"first_year": "Year", "customer_id": "new"})
)

orders_plot = orders[orders["Year"].between(2013, 2022)]
active = (
    orders_plot.groupby("Year")["customer_id"]
    .nunique()
    .reset_index()
    .rename(columns={"customer_id": "active"})
)

df_plot = active.merge(new_custs, on="Year", how="left").fillna(0)
years_arr = df_plot["Year"].values
active_arr = df_plot["active"].values
new_arr    = df_plot["new"].values

# -- Data for Right Chart (Cohort Retention) --
print("Đang tính toán ma trận Cohort Retention...")
orders2 = orders.merge(first_yr.rename('cohort'), on='customer_id')
COHORT_YEARS = list(range(2012, 2022))

cohort_data = {}
for c_yr in COHORT_YEARS:
    c_custs = set(orders2[orders2['cohort'] == c_yr]['customer_id'])
    row = {}
    for delta in range(8):
        yr = c_yr + delta
        if yr <= 2022:
            n = orders[(orders['Year'] == yr) &
                       (orders['customer_id'].isin(c_custs))]['customer_id'].nunique()
            row[delta] = n / len(c_custs) * 100 if c_custs else np.nan
        else:
            row[delta] = np.nan
    cohort_data[c_yr] = row

cohort_df = pd.DataFrame(cohort_data).T

# ── 2. Vẽ Biểu Đồ ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
fig.patch.set_facecolor("white")

# ==============================================================================
# CHART 1 (LEFT): Active & New Customers
# ==============================================================================
ax1.set_facecolor("white")

# Line: Active customers
ax1.plot(years_arr, active_arr / 1e3, color=COLOR_LINE, linewidth=2,
         linestyle="-", zorder=4)
ax1.scatter(years_arr, active_arr / 1e3, color=COLOR_DOT, s=48, zorder=5)

# Data labels trên dot
for yr, val in zip(years_arr, active_arr):
    offset = max(active_arr) / 1e3 * 0.04
    ax1.text(yr, val / 1e3 + offset,
             f"{val/1e3:.1f}K",
             ha="center", va="bottom",
             fontsize=8, color=COLOR_DOT, fontweight="500")

# Bars: New customers
bars = ax1.bar(years_arr, new_arr / 1e3, color=COLOR_BAR, width=0.6,
               zorder=2, linewidth=0, alpha=0.85)

# Data labels trên bar
for yr, val in zip(years_arr, new_arr):
    ax1.text(yr, val / 1e3 + max(active_arr) / 1e3 * 0.02,
             f"{val/1e3:.1f}K",
             ha="center", va="bottom",
             fontsize=8, color=LABEL_COLOR, fontweight="500")

# Trục Y: Customers
y_max1     = max(active_arr) / 1e3
tick_step1 = 5 if y_max1 <= 50 else 10
upper1     = np.ceil(y_max1 / tick_step1) * tick_step1
ticks1     = np.arange(0, upper1 + tick_step1 / 2, tick_step1)

ax1.set_yticks(ticks1)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v:.0f}K" if v > 0 else "0")
)
ax1.set_ylim(0, ticks1[-1])
ax1.tick_params(axis="y", labelsize=9, colors=TICK_COLOR)
ax1.set_ylabel("Customers (K)", fontsize=10,
               color=AXIS_COLOR, labelpad=10, fontweight="500")

# Trục X
ax1.set_xlim(years_arr[0] - 0.6, years_arr[-1] + 0.6)
ax1.set_xticks(years_arr)
ax1.set_xticklabels(years_arr, fontsize=9, color=TICK_COLOR)
ax1.tick_params(axis="x", length=0)
ax1.set_xlabel("Năm", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight="500")

# Grid & spines
ax1.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["bottom"].set_color(GRID_COLOR)

# Đường đóng khung trái/phải
ax1.axvline(years_arr[0]  - 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)
ax1.axvline(years_arr[-1] + 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)

# Legend
legend_elements = [
    plt.Line2D([0], [0], color=COLOR_LINE, linewidth=2,
               marker="o", markersize=5, markerfacecolor=COLOR_DOT,
               label="Active customers"),
    mpatches.Patch(facecolor=COLOR_BAR, alpha=0.85, label="New customers"),
]
ax1.legend(handles=legend_elements, loc="upper right",
           frameon=True, framealpha=0.9, edgecolor=GRID_COLOR,
           fontsize=9, handlelength=1.5)

# Title

# ==============================================================================
# CHART 2 (RIGHT): Cohort Retention Heatmap
# ==============================================================================
ax2.set_facecolor("white")

cmap_hm = LinearSegmentedColormap.from_list(
    'retention',
    ['#E74C3C', '#F39C12', '#F1C40F', '#27AE60'], 
    N=256
)

data_hm = cohort_df.values
im = ax2.imshow(data_hm, aspect='auto', cmap=cmap_hm, vmin=0, vmax=75)

# Text labels
for i in range(len(COHORT_YEARS)):
    for j in range(8):
        v = data_hm[i, j]
        if not np.isnan(v):
            text_color = 'white' if (v < 25 or v > 55) else '#222222'
            ax2.text(j, i, f'{v:.0f}%', ha='center', va='center',
                     fontsize=9.5, fontweight='500', color=text_color)

# Trục X & Y
ax2.set_xticks(range(8))
ax2.set_xticklabels([f'Yr + {d}' for d in range(8)], fontsize=9, color=TICK_COLOR)
ax2.set_yticks(range(len(COHORT_YEARS)))
ax2.set_yticklabels([str(c) for c in COHORT_YEARS], fontsize=9, color=TICK_COLOR)

ax2.set_xlabel("Số năm kể từ lần đầu mua hàng", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight="500")
ax2.set_ylabel("Năm Cohort (Năm đầu mua hàng)", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight="500")

ax2.tick_params(axis="both", length=0)

# Spines
for spine in ax2.spines.values():
    spine.set_visible(False)

# Colorbar
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Retention %', color=AXIS_COLOR, fontsize=10, fontweight="500", labelpad=10)
cbar.ax.tick_params(labelsize=9, colors=TICK_COLOR, length=0)
cbar.outline.set_visible(False)

# Title

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=3.0)
plt.savefig(
    "reports/figures/merge_2_combined.png",
    dpi=180, bbox_inches="tight", facecolor="white",
)
plt.show()
print("Saved → merge_2_combined.png")
