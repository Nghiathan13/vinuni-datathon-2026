import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── Shared style ──────────────────────────────────────────────────────────────
COLOR_BAR    = "#2D7D46"   # new customers — bar (xanh)
COLOR_LINE   = "#E8B84B"   # active customers — line (vàng)
COLOR_DOT    = "#C8960A"   # dot trên line (vàng đậm)
GRID_COLOR   = "#EBEBEB"
LABEL_COLOR  = "#222222"
AXIS_COLOR   = "#444444"
TICK_COLOR   = "#555555"

# ── Data ──────────────────────────────────────────────────────────────────────
orders = pd.read_csv(
    "data/orders.csv",
    parse_dates=["order_date"],
)
orders["Year"] = orders["order_date"].dt.year

# New customers: năm đặt đơn đầu tiên của mỗi khách (tính trên toàn bộ dữ liệu, bao gồm cả 2012)
first_order_year = (
    orders.groupby("customer_id")["Year"]
    .min()
    .reset_index()
    .rename(columns={"Year": "first_year"})
)
new_custs = (
    first_order_year.groupby("first_year")["customer_id"]
    .count()
    .reset_index()
    .rename(columns={"first_year": "Year", "customer_id": "new"})
)

# Active customers: lọc từ 2013-2022 cho biểu đồ
orders_plot = orders[orders["Year"].between(2013, 2022)]
active = (
    orders_plot.groupby("Year")["customer_id"]
    .nunique()
    .reset_index()
    .rename(columns={"customer_id": "active"})
)

df_plot = active.merge(new_custs, on="Year", how="left").fillna(0)
years   = df_plot["Year"].values
active  = df_plot["active"].values
new     = df_plot["new"].values

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# ── Line: Active customers ────────────────────────────────────────────────────
ax1.plot(years, active / 1e3, color=COLOR_LINE, linewidth=2,
         linestyle="-", zorder=4)
ax1.scatter(years, active / 1e3, color=COLOR_DOT, s=48, zorder=5)

# Data labels trên dot
for yr, val in zip(years, active):
    offset = max(active) / 1e3 * 0.04
    ax1.text(yr, val / 1e3 + offset,
             f"{val/1e3:.1f}K",
             ha="center", va="bottom",
             fontsize=8, color=COLOR_DOT, fontweight="500")

# ── Bars: New customers ───────────────────────────────────────────────────────
bars = ax1.bar(years, new / 1e3, color=COLOR_BAR, width=0.6,
               zorder=2, linewidth=0, alpha=0.85)

# Data labels trên bar
for yr, val in zip(years, new):
    ax1.text(yr, val / 1e3 + max(active) / 1e3 * 0.02,
             f"{val/1e3:.1f}K",
             ha="center", va="bottom",
             fontsize=8, color=LABEL_COLOR, fontweight="500")

# ── Trục Y: Customers ─────────────────────────────────────────────────────────
y_max1     = max(active) / 1e3
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
               color=AXIS_COLOR, labelpad=10, fontweight=500)

# ── Trục X ────────────────────────────────────────────────────────────────────
ax1.set_xlim(years[0] - 0.6, years[-1] + 0.6)
ax1.set_xticks(years)
ax1.set_xticklabels(years, fontsize=9, color="#333333")
ax1.tick_params(axis="x", length=0)
ax1.set_xlabel("Năm", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight=500)

# ── Grid & spines ─────────────────────────────────────────────────────────────
ax1.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["bottom"].set_color(GRID_COLOR)

# Đường đóng khung trái/phải
ax1.axvline(years[0]  - 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)
ax1.axvline(years[-1] + 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)



# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    plt.Line2D([0], [0], color=COLOR_LINE, linewidth=2,
               marker="o", markersize=5, markerfacecolor=COLOR_DOT,
               label="Active customers"),
    mpatches.Patch(facecolor=COLOR_BAR, alpha=0.85, label="New customers"),
]
ax1.legend(handles=legend_elements, loc="upper right",
           frameon=True, framealpha=0.9, edgecolor=GRID_COLOR,
           fontsize=9, handlelength=1.5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax1.set_title("Active & New Customers theo Năm (2013–2022)",
              fontsize=13, fontweight="bold", color="#1A1A1A", pad=14)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(
    "reports/figures/active_new_customers.png",
    dpi=180, bbox_inches="tight", facecolor="white",
)
plt.show()
print("Saved → active_new_customers.png")
