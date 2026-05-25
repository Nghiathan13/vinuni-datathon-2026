import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.ticker as mticker
# Cài đặt style cho đồ thị
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. ĐỌC DỮ LIỆU & HUẤN LUYỆN MODEL THEO THÁNG (2012-2022)
# ==========================================
# Đọc file orders.csv
orders = pd.read_csv('data/orders.csv', parse_dates=['order_date'])

# Lấy ngày mua hàng đầu tiên của mỗi khách hàng
first_purchases = orders.groupby('customer_id')['order_date'].min().reset_index()
first_purchases.columns = ['customer_id', 'first_order_date']

# Gom nhóm đếm số khách hàng mới theo từng tháng
first_purchases['month_year'] = first_purchases['first_order_date'].dt.to_period('M')
df_monthly = first_purchases.groupby('month_year')['customer_id'].nunique().reset_index()
df_monthly.columns =['Date', 'New_Customers']

# Chuyển Date về Timestamp
df_monthly['Date'] = df_monthly['Date'].dt.to_timestamp()
df_monthly.set_index('Date', inplace=True)
df_monthly = df_monthly.resample('MS').sum().fillna(0)

# Xử lý số âm bằng Log Transformation
df_monthly['Log_New_Customers'] = np.log1p(df_monthly['New_Customers'])

# Chạy mô hình Holt-Winters trên chuỗi Log (Học từ toàn bộ 2012 - 2022)
model = ExponentialSmoothing(df_monthly['Log_New_Customers'], 
                             trend='add', seasonal='add', seasonal_periods=12, damped_trend=True)
fit_model = model.fit()

# Dự báo 24 tháng (2023 - 2024)
forecast_steps = 24
log_forecast = fit_model.forecast(forecast_steps)
# Khôi phục số liệu thực (khử Log)
forecast_monthly_values = np.round(np.maximum(np.expm1(log_forecast), 0))
forecast_dates = pd.date_range(start='2023-01-01', periods=forecast_steps, freq='MS')

# ==========================================
# 2. CHUYỂN ĐỔI SANG DỮ LIỆU NĂM ĐỂ VẼ BIỂU ĐỒ (2013-2024)
# ==========================================
# Tính tổng khách hàng thực tế theo NĂM
df_monthly['Year'] = df_monthly.index.year
df_yearly_actual = df_monthly.groupby('Year')['New_Customers'].sum().reset_index()

# CHỈ lấy dữ liệu từ 2013 trở đi để vẽ biểu đồ (ẩn 2012 đi)
df_yearly_actual = df_yearly_actual[df_yearly_actual['Year'] >= 2013]

# Tính tổng khách hàng dự báo theo NĂM (2023, 2024)
df_forecast_monthly = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_monthly_values})
df_forecast_monthly['Year'] = df_forecast_monthly['Date'].dt.year
df_yearly_forecast = df_forecast_monthly.groupby('Year')['Forecast'].sum().reset_index()

# THỦ THUẬT NỐI NÉT ĐỨT: Lấy điểm 2022 chèn vào mảng dự báo để đường vẽ liền mạch
val_2022 = df_yearly_actual.loc[df_yearly_actual['Year'] == 2022, 'New_Customers'].values[0]
df_connect = pd.DataFrame({'Year':[2022], 'Forecast': [val_2022]})
df_yearly_forecast_plot = pd.concat([df_connect, df_yearly_forecast], ignore_index=True)

# ==========================================
# 3. TRỰC QUAN HÓA (Style đồng bộ active_new_customer.py)
# ==========================================
# ── Shared style ──────────────────────────────────────────────────────────────
COLOR_ACTUAL_LINE = "#2D7D46"
COLOR_ACTUAL_DOT  = "#1B5E30"
COLOR_FCST_LINE   = "#E8B84B"
COLOR_FCST_DOT    = "#C8960A"
GRID_COLOR   = "#EBEBEB"
LABEL_COLOR  = "#222222"
AXIS_COLOR   = "#444444"
TICK_COLOR   = "#555555"

years_actual = df_yearly_actual['Year'].values
actual_vals  = df_yearly_actual['New_Customers'].values
years_fcst   = df_yearly_forecast_plot['Year'].values
fcst_vals    = df_yearly_forecast_plot['Forecast'].values

all_years = np.concatenate([years_actual, df_yearly_forecast['Year'].values])
max_val = max(np.max(actual_vals), np.max(fcst_vals))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# ── Line: Thực tế ─────────────────────────────────────────────────────────────
ax1.plot(years_actual, actual_vals / 1e3, color=COLOR_ACTUAL_LINE, linewidth=2,
         linestyle="-", zorder=4)
ax1.scatter(years_actual, actual_vals / 1e3, color=COLOR_ACTUAL_DOT, s=48, zorder=5)

# Data labels trên dot thực tế
for yr, val in zip(years_actual, actual_vals):
    offset = max_val / 1e3 * 0.04
    ax1.text(yr, val / 1e3 + offset,
             f"{val/1e3:.1f}K",
             ha="center", va="bottom",
             fontsize=8, color=LABEL_COLOR, fontweight="bold")

# ── Line: Dự báo ──────────────────────────────────────────────────────────────
ax1.plot(years_fcst, fcst_vals / 1e3, color=COLOR_FCST_LINE, linewidth=2,
         linestyle="--", zorder=4)
ax1.scatter(years_fcst[1:], fcst_vals[1:] / 1e3, color=COLOR_FCST_DOT, s=48, zorder=5)

# Data labels trên dot dự báo (bỏ qua điểm nối năm 2022)
for yr, val in zip(df_yearly_forecast['Year'], df_yearly_forecast['Forecast']):
    offset = max_val / 1e3 * 0.04
    ax1.text(yr, val / 1e3 + offset,
             f"{val/1e3:.1f}K",
             ha="center", va="bottom",
             fontsize=8, color=COLOR_FCST_DOT, fontweight="bold")

# ── Trục Y ────────────────────────────────────────────────────────────────────
y_max1     = max_val / 1e3
tick_step1 = 5 if y_max1 <= 50 else 10
upper1     = np.ceil(y_max1 / tick_step1) * tick_step1
ticks1     = np.arange(0, upper1 + tick_step1 / 2, tick_step1)

ax1.set_yticks(ticks1)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v:.0f}K" if v > 0 else "0")
)
ax1.set_ylim(0, ticks1[-1])
ax1.tick_params(axis="y", labelsize=9, colors=TICK_COLOR)
ax1.set_ylabel("Khách hàng mới (K)", fontsize=10,
               color=AXIS_COLOR, labelpad=10, fontweight=500)

# ── Trục X ────────────────────────────────────────────────────────────────────
ax1.set_xlim(all_years[0] - 0.6, all_years[-1] + 0.6)
ax1.set_xticks(all_years)
ax1.set_xticklabels(all_years, fontsize=9, color="#333333")
ax1.tick_params(axis="x", length=0)
ax1.set_xlabel("Năm", fontsize=10, color=AXIS_COLOR, labelpad=10, fontweight=500)

# ── Grid & spines ─────────────────────────────────────────────────────────────
ax1.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
ax1.xaxis.grid(False) # Tắt lưới dọc ở giữa
ax1.set_axisbelow(True)
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["bottom"].set_color(GRID_COLOR)

# Đường đóng khung trái/phải
ax1.axvline(all_years[0]  - 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)
ax1.axvline(all_years[-1] + 0.6, color=GRID_COLOR, linewidth=0.8, zorder=1)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    plt.Line2D([0], [0], color=COLOR_ACTUAL_LINE, linewidth=2,
               marker="o", markersize=5, markerfacecolor=COLOR_ACTUAL_DOT,
               label="Thực tế (2013-2022)"),
    plt.Line2D([0], [0], color=COLOR_FCST_LINE, linewidth=2, linestyle="--",
               marker="o", markersize=5, markerfacecolor=COLOR_FCST_DOT,
               label="Dự báo (2023-2024)"),
]
ax1.legend(handles=legend_elements, loc="upper right",
           frameon=True, framealpha=0.9, edgecolor=GRID_COLOR,
           fontsize=9, handlelength=2)

# ── Title & Subtitle ──────────────────────────────────────────────────────────

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(
    "reports/figures/forecast_yearly_clean.png",
    dpi=180, bbox_inches="tight", facecolor="white",
)
plt.show()

# In kết quả dự báo ra terminal
print("-" * 30)
for index, row in df_yearly_forecast.iterrows():
    print(f"Dự báo khách hàng mới năm {int(row['Year'])}: {int(row['Forecast']):,} người")
print("-" * 30)
