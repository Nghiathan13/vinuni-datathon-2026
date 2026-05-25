import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── Shared style ──────────────────────────────────────────────────────────────
LABEL_COLOR  = "#222222"
AXIS_COLOR   = "#444444"
TICK_COLOR   = "#555555"

# ── Data ──────────────────────────────────────────────────────────────────────
orders = pd.read_csv(
    "data/orders.csv",
    parse_dates=["order_date"],
)
orders["year"] = orders["order_date"].dt.year
first_yr = orders.groupby("customer_id")["year"].min()

orders2 = orders.merge(first_yr.rename('cohort'), on='customer_id')
COHORT_YEARS = list(range(2012, 2022))

print("Đang tính toán ma trận Cohort Retention...")
cohort_data = {}
for c_yr in COHORT_YEARS:
    c_custs = set(orders2[orders2['cohort'] == c_yr]['customer_id'])
    row = {}
    for delta in range(8):
        yr = c_yr + delta
        if yr <= 2022:
            n = orders[(orders['year'] == yr) &
                       (orders['customer_id'].isin(c_custs))]['customer_id'].nunique()
            row[delta] = n / len(c_custs) * 100 if c_custs else np.nan
        else:
            row[delta] = np.nan
    cohort_data[c_yr] = row

cohort_df = pd.DataFrame(cohort_data).T

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Colormap phù hợp cho nền trắng (Đỏ -> Cam -> Vàng -> Xanh lá)
cmap_hm = LinearSegmentedColormap.from_list(
    'retention',
    ['#E74C3C', '#F39C12', '#F1C40F', '#27AE60'], 
    N=256
)

data_hm = cohort_df.values
# Thêm vmin và vmax để giữ phổ màu chuẩn
im = ax.imshow(data_hm, aspect='auto', cmap=cmap_hm, vmin=0, vmax=75)

# Text labels trên ô
for i in range(len(COHORT_YEARS)):
    for j in range(8):
        v = data_hm[i, j]
        if not np.isnan(v):
            # Chọn màu text tương phản với màu ô
            text_color = 'white' if (v < 25 or v > 55) else '#222222'
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                    fontsize=9.5, fontweight='500', color=text_color)

# ── Trục X & Y ────────────────────────────────────────────────────────────────
ax.set_xticks(range(8))
ax.set_xticklabels([f'Yr + {d}' for d in range(8)], fontsize=9, color="#333333")
ax.set_yticks(range(len(COHORT_YEARS)))
ax.set_yticklabels([str(c) for c in COHORT_YEARS], fontsize=9, color="#333333")

ax.set_xlabel("Số năm kể từ lần đầu mua hàng", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight="500")
ax.set_ylabel("Năm Cohort (Năm đầu mua hàng)", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight="500")

ax.tick_params(axis="both", length=0) # Ẩn gạch chia độ

# ── Spines (Viền) ─────────────────────────────────────────────────────────────
for spine in ax.spines.values():
    spine.set_visible(False)



# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Retention %', color=AXIS_COLOR, fontsize=10, fontweight="500", labelpad=10)
cbar.ax.tick_params(labelsize=9, colors=TICK_COLOR, length=0)
cbar.outline.set_visible(False)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("Cohort Retention Heatmap (% khách còn mua hàng sau N năm)",
              fontsize=13, fontweight="bold", color="#1A1A1A", pad=14)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(
    "reports/figures/cohort_retention_heatmap.png",
    dpi=180, bbox_inches="tight", facecolor="white",
)
plt.show()
print("Saved → cohort_retention_heatmap.png")
