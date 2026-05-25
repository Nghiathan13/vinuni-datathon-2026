# 850.000



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("1. Loading data...")
DATA_DIR = 'data/'
sales = pd.read_csv(DATA_DIR + 'sales.csv', parse_dates=['Date'])
test = pd.read_csv(DATA_DIR + 'sample_submission.csv', parse_dates=['Date'])
promos = pd.read_csv(DATA_DIR + 'promotions.csv', parse_dates=['start_date', 'end_date'])
traffic = pd.read_csv(DATA_DIR + 'web_traffic.csv', parse_dates=['date'])
orders = pd.read_csv(DATA_DIR + 'orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(DATA_DIR + 'order_items.csv')

# ==========================================
# 2. BEHAVIORAL FEATURES (Model 10 & 29 Heritage)
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

# Monthly AOV & Margin (Key from Model 29)
order_values = order_items.groupby('order_id')['unit_price'].sum().reset_index()
orders_with_val = pd.merge(orders, order_values, on='order_id')
avg_aov = orders_with_val[orders_with_val['year'] >= 2019].groupby('month')['unit_price'].mean().to_dict()

sales['margin_rate'] = (sales['Revenue'] - sales['COGS']) / sales['Revenue']
avg_margin = sales[sales['Date'].dt.year >= 2019].groupby(sales['Date'].dt.month)['margin_rate'].mean().to_dict()

# ==========================================
# 3. TREND (Model 10 Precise Logic)
# ==========================================
print("3. Calculating Trends (Model 10 Style)...")
sales['year'] = sales['Date'].dt.year
annual = sales.groupby('year')[['Revenue', 'COGS']].sum()
recent_years = annual.loc[2020:2022]

growth_rev = (1 + recent_years['Revenue'].pct_change().dropna()).prod() ** (1/2)
growth_cogs = (1 + recent_years['COGS'].pct_change().dropna()).prod() ** (1/2)

print(f"Growth Rates: Rev {growth_rev:.4f}, COGS {growth_cogs:.4f}")

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
    df['feat_conv'] = df['month'].map(avg_conv)
    df['feat_aov'] = df['month'].map(avg_aov)
    df['feat_margin'] = df['month'].map(avg_margin)
    
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
# 5. TRAINING (Model 10 Style Regression)
# ==========================================
print("5. Training Model 34 (L2 Regression)...")
# Tuyến tính (Model 10) với Boost 2019+
train_features['weight'] = 1.0 + (train_features['Date'].dt.year - 2012) / 10.0
train_features.loc[train_features['Date'].dt.year >= 2019, 'weight'] *= 1.5

val_mask = train_features['Date'].dt.year == 2022
X_train = train_features.loc[~val_mask, FEATURES]
y_rev_train = train_features.loc[~val_mask, 'rev_norm']
y_cogs_train = train_features.loc[~val_mask, 'cogs_norm']
w_train = train_features.loc[~val_mask, 'weight']

X_val = train_features.loc[val_mask, FEATURES]
y_rev_val = train_features.loc[val_mask, 'rev_norm']
y_cogs_val = train_features.loc[val_mask, 'cogs_norm']

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'n_estimators': 2500,
    'learning_rate': 0.015,
    'max_depth': 7,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1
}

model_rev = lgb.LGBMRegressor(**lgb_params)
model_rev.fit(X_train, y_rev_train, sample_weight=w_train, eval_set=[(X_val, y_rev_val)], callbacks=[lgb.early_stopping(150)])

model_cogs = lgb.LGBMRegressor(**lgb_params)
model_cogs.fit(X_train, y_cogs_train, sample_weight=w_train, eval_set=[(X_val, y_cogs_val)], callbacks=[lgb.early_stopping(150)])

# ==========================================
# 6. EVAL & SUBMISSION
# ==========================================
val_results = train_features.loc[val_mask].copy()
val_results['Rev_Pred'] = model_rev.predict(X_val) * val_results['trend_rev']
val_results['COGS_Pred'] = model_cogs.predict(X_val) * val_results['trend_cogs']

print("\n" + "="*50)
print(" VALIDATION RESULTS (2022)")
print("="*50)
rev_mae = mean_absolute_error(val_results['Revenue'], val_results['Rev_Pred'])
cogs_mae = mean_absolute_error(val_results['COGS'], val_results['COGS_Pred'])
print(f"Revenue MAE: {rev_mae:.2f}")
print(f"COGS MAE   : {cogs_mae:.2f}")
print(f"Total MAE  : {(rev_mae + cogs_mae)/2:.2f}")

X_test = test_features[FEATURES]
test_features['Revenue'] = (model_rev.predict(X_test) * test_features['trend_rev']).round(2)
test_features['COGS'] = (model_cogs.predict(X_test) * test_features['trend_cogs']).round(2)

# Safety clip lỏng lẻo hơn để cho phép biên lợi nhuận âm (tháng 8, 12)
test_features['COGS'] = np.maximum(test_features['COGS'], 0)
test_features['COGS'] = np.minimum(test_features['COGS'], test_features['Revenue'] * 1.3) # Cho phép lỗ tối đa 30%

submission = test_features[['Date', 'Revenue', 'COGS']]
submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')
submission.to_csv('submission.csv', index=False)
print("\n[+] Success! Model 34 saved to submission.csv")
