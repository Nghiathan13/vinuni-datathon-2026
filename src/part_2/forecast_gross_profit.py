import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("1. Loading data...")
DATA_DIR = 'data/'
sales = pd.read_csv(DATA_DIR + 'sales.csv', parse_dates=['Date'])
promos = pd.read_csv(DATA_DIR + 'promotions.csv', parse_dates=['start_date', 'end_date'])
traffic = pd.read_csv(DATA_DIR + 'web_traffic.csv', parse_dates=['date'])
orders = pd.read_csv(DATA_DIR + 'orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(DATA_DIR + 'order_items.csv')

# Create forecast dates for 2023 and 2024
forecast_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
test = pd.DataFrame({'Date': forecast_dates})

# ==========================================
# 2. BEHAVIORAL FEATURES
# ==========================================
print("2. Calculating Behavioral Patterns...")
traffic['year'], traffic['month'] = traffic['date'].dt.year, traffic['date'].dt.month
orders['year'], orders['month'] = orders['order_date'].dt.year, orders['order_date'].dt.month

# Monthly Conversion
m_stats = pd.merge(
    traffic.groupby(['year', 'month'])['sessions'].sum().reset_index(),
    orders.groupby(['year', 'month'])['order_id'].count().reset_index(),
    on=['year', 'month']
)
m_stats['conv'] = m_stats['order_id'] / m_stats['sessions']
avg_conv = m_stats[m_stats['year'] >= 2019].groupby('month')['conv'].mean().to_dict()

# Monthly AOV
order_values = order_items.groupby('order_id')['unit_price'].sum().reset_index()
orders_with_val = pd.merge(orders, order_values, on='order_id')
avg_aov = orders_with_val[orders_with_val['year'] >= 2019].groupby('month')['unit_price'].mean().to_dict()

# Monthly Margin
sales['margin_rate'] = (sales['Revenue'] - sales['COGS']) / sales['Revenue']
avg_margin = sales[sales['Date'].dt.year >= 2019].groupby(sales['Date'].dt.month)['margin_rate'].mean().to_dict()

# ==========================================
# 3. TREND (Model 10 Style)
# ==========================================
print("3. Calculating Trends...")
sales['year'] = sales['Date'].dt.year
annual = sales.groupby('year')[['Revenue', 'COGS']].sum()
recent_years = annual.loc[2020:2022]

growth_rev = (1 + recent_years['Revenue'].pct_change().dropna()).prod() ** (1/2)
growth_cogs = (1 + recent_years['COGS'].pct_change().dropna()).prod() ** (1/2)

base_rev_daily = annual.loc[2022, 'Revenue'] / 365
base_cogs_daily = annual.loc[2022, 'COGS'] / 365

def apply_trend(df):
    df = df.copy()
    df['years_ahead'] = df['Date'].dt.year - 2022
    df['trend_rev'] = base_rev_daily * (growth_rev ** df['years_ahead'])
    df['trend_cogs'] = base_cogs_daily * (growth_cogs ** df['years_ahead'])
    return df

train = apply_trend(sales)
test_df = apply_trend(test)
train['rev_norm'] = train['Revenue'] / train['trend_rev']
train['cogs_norm'] = train['COGS'] / train['trend_cogs']

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
def create_features(df, promos_df):
    df = df.copy()
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dow'] = df['Date'].dt.dayofweek
    df['doy'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    
    # Cyclical
    df['sin_doy'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['doy'] / 365.25)
    
    # Behavioral
    df['feat_conv'] = df['month'].map(avg_conv).ffill().bfill()
    df['feat_aov'] = df['month'].map(avg_aov).ffill().bfill()
    df['feat_margin'] = df['month'].map(avg_margin).ffill().bfill()
    
    # Indicators
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['post_2019'] = (df['Date'].dt.year >= 2019).astype(int)
    
    # Promo
    p_counts = []
    for d in df['Date']:
        p_counts.append(len(promos_df[(promos_df['start_date'] <= d) & (promos_df['end_date'] >= d)]))
    df['promo_count'] = p_counts
    
    return df

train_features = create_features(train, promos)
test_features = create_features(test_df, promos)

FEATURES = ['month', 'day', 'dow', 'doy', 'quarter', 'sin_doy', 'cos_doy',
            'feat_conv', 'feat_aov', 'feat_margin', 'is_weekend', 'post_2019', 'promo_count']

# ==========================================
# 5. TRAINING
# ==========================================
print("5. Training Models...")
train_features['weight'] = 1.0 + (train_features['Date'].dt.year - 2012) / 10.0
train_features.loc[train_features['Date'].dt.year >= 2019, 'weight'] *= 1.5

X_train = train_features[FEATURES]
y_rev_train = train_features['rev_norm']
y_cogs_train = train_features['cogs_norm']
w_train = train_features['weight']

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1
}

model_rev = lgb.LGBMRegressor(**lgb_params)
model_rev.fit(X_train, y_rev_train, sample_weight=w_train)

model_cogs = lgb.LGBMRegressor(**lgb_params)
model_cogs.fit(X_train, y_cogs_train, sample_weight=w_train)

# ==========================================
# 6. FORECASTING
# ==========================================
print("6. Forecasting 2023-2024...")
X_test = test_features[FEATURES]
test_features['Revenue'] = (model_rev.predict(X_test) * test_features['trend_rev'])
test_features['COGS'] = (model_cogs.predict(X_test) * test_features['trend_cogs'])
test_features['GrossProfit'] = test_features['Revenue'] - test_features['COGS']

# ==========================================
# 7. VISUALIZATION
# ==========================================
print("7. Generating Visualization...")
# Historical Gross Profit
sales['GrossProfit'] = sales['Revenue'] - sales['COGS']
historical_annual = sales.groupby('year')['GrossProfit'].sum().reset_index()

# Forecasted Gross Profit
test_features['year'] = test_features['Date'].dt.year
forecast_annual = test_features.groupby('year')['GrossProfit'].sum().reset_index()

# Combine for plotting
combined_annual = pd.concat([historical_annual, forecast_annual]).reset_index(drop=True)

# Plotting style
BAR_COLOR_HIST = "#A8D5B5" # Nhạt cho lịch sử
BAR_COLOR_FORE = "#2D7D46" # Đậm cho dự báo
GRID_COLOR = "#EBEBEB"

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

years = combined_annual['year'].values
values = combined_annual['GrossProfit'].values
colors = [BAR_COLOR_HIST if y <= 2022 else BAR_COLOR_FORE for y in years]

bars = ax.bar(years, values, color=colors, width=0.7, zorder=2)

# Labels
for yr, val in zip(years, values):
    ax.text(yr, val + max(values) * 0.02, f"{val/1e9:.2f}B", 
            ha='center', va='bottom', fontsize=9, fontweight='bold' if yr > 2022 else 'normal')

ax.set_title("Dự báo Lợi nhuận gộp 2023-2024", fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel("Lợi nhuận gộp (Tỷ USD)", fontsize=10)
ax.set_xlabel("Năm", fontsize=10)

# Format y-axis to Billions
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1e9:.1f}B'))
ax.grid(axis='y', linestyle='--', alpha=0.7, color=GRID_COLOR, zorder=0)

plt.tight_layout()
plt.savefig("reports/figures/forecast_gross_profit_2023_2024.png", dpi=300)
print("Visualization saved as forecast_gross_profit_2023_2024.png")

# Output yearly totals
for _, row in forecast_annual.iterrows():
    print(f"Năm {int(row['year'])}: Lợi nhuận gộp dự báo = {row['GrossProfit']:,.2f}")

# Save the forecast code to part_2/1.py as requested
# (The code above is the code for part_2/1.py)
