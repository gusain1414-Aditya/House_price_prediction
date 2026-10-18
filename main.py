import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Output files are written next to this script (override with the OUT_DIR env var)
OUT_DIR = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)
PNG_PATH = os.path.join(OUT_DIR, "ml_analysis.png")
JSON_PATH = os.path.join(OUT_DIR, "model_results.json")
JS_PATH = os.path.join(OUT_DIR, "dashboard_data.js")    # loaded by dashboard.html
CSV_PATH = os.path.join(OUT_DIR, "house_price_data.csv")

# =============================================================
# 1. DATA LOADING & EXPLORATION
# =============================================================
print("=" * 60)
print("  HOUSE PRICE PREDICTION — ML PIPELINE")
print("=" * 60)

# Synthetic California-style housing dataset (mirrors real dataset structure)
np.random.seed(42)
n = 8000
lat = np.random.uniform(32.5, 41.9, n)
lon = np.random.uniform(-124.3, -114.3, n)       # California longitudes are negative
age = np.random.randint(1, 52, n).astype(float)
rooms = np.random.uniform(2.5, 10.0, n)
bedrms = np.clip(rooms * 0.3 + np.random.normal(0, 0.4, n), 1.0, rooms - 0.5)
occup = np.random.uniform(1.0, 6.0, n)
pop = np.random.randint(80, 5000, n).astype(float)
inc = np.random.uniform(0.5, 15.0, n)

# Realistic price formula
price = (
    inc * 0.42
    + rooms * 0.06
    - age * 0.006
    - occup * 0.05
    + (lat - 34) * 0.04
    - (lon + 120) * 0.02
    + np.random.normal(0, 0.28, n)
).clip(0.45, 5.0)

df = pd.DataFrame({
    'MedInc': inc, 'HouseAge': age, 'AveRooms': rooms,
    'AveBedrms': bedrms, 'Population': pop, 'AveOccup': occup,
    'Latitude': lat, 'Longitude': lon, 'Price': price
})

print(f"\n[1] Dataset: California Housing (synthetic)  |  Shape: {df.shape}")
print(f"    Features: {list(df.columns)}")
print(f"\n    Stats:\n{df.describe().round(2).to_string()}")

# =============================================================
# 2. DATA PREPROCESSING
# =============================================================
print("\n[2] Preprocessing...")
# Inject 2% missing values to demonstrate imputation
np.random.seed(42)
for col in ['MedInc', 'AveRooms', 'AveBedrms']:
    idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[idx, col] = np.nan
missing_injected = int(df.isnull().sum().sum())
print(f"    Missing values injected: {missing_injected} total")

# Impute
imputer = SimpleImputer(strategy='median')
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print(f"    After imputation: {df.isnull().sum().sum()} missing values")

# =============================================================
# 3. FEATURE ENGINEERING
# =============================================================
print("\n[3] Feature Engineering...")
df['RoomsPerPerson'] = df['AveRooms'] / df['AveOccup'].clip(lower=1)
df['BedroomRatio'] = df['AveBedrms'] / df['AveRooms'].clip(lower=1)
df['PopDensity'] = df['Population'] / df['HouseAge'].clip(lower=1)
df['IncomePerRoom'] = df['MedInc'] / df['AveRooms'].clip(lower=1)
df['LocationScore'] = np.sqrt(df['Latitude'] ** 2 + df['Longitude'] ** 2)

new_features = ['RoomsPerPerson', 'BedroomRatio', 'PopDensity',
                'IncomePerRoom', 'LocationScore']
print(f"    New features created: {new_features}")

# =============================================================
# 4. TRAIN / TEST SPLIT & SCALING
# =============================================================
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
print(f"\n[4] Split — Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# =============================================================
# 5. MODEL TRAINING & EVALUATION
# =============================================================
print("\n[5] Training Models...")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    cv = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='r2').mean()
    results[name] = {'RMSE': round(rmse, 4), 'R2': round(r2, 4),
                     'MAE': round(mae, 4), 'CV_R2': round(cv, 4)}
    print(f"    {name:<25} RMSE={rmse:.4f}  R²={r2:.4f}  CV R²={cv:.4f}")

# =============================================================
# 6. HYPERPARAMETER TUNING (Best Model)
# =============================================================
print("\n[6] Hyperparameter Tuning — Gradient Boosting...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
}
gs = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
)
gs.fit(X_train_sc, y_train)
best = gs.best_estimator_
best_preds = best.predict(X_test_sc)
best_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
best_r2 = r2_score(y_test, best_preds)
print(f"    Best params : {gs.best_params_}")
print(f"    Tuned RMSE  : {best_rmse:.4f}")
print(f"    Tuned R²    : {best_r2:.4f}")

results["Gradient Boosting Tuned"] = {
    'RMSE': round(best_rmse, 4),
    'R2': round(best_r2, 4),
    'MAE': round(mean_absolute_error(y_test, best_preds), 4),
    'CV_R2': round(gs.best_score_, 4),
}

# =============================================================
# 7. FEATURE IMPORTANCE
# =============================================================
fi = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)

# =============================================================
# 8. VISUALIZATIONS
# =============================================================
BG = "#0d1117"
CARD = "#161b22"
ACC = "#58a6ff"
GRN = "#3fb950"
ORG = "#f78166"
PRP = "#bc8cff"
TXT = "#e6edf3"
GRY = "#30363d"

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'text.color': TXT, 'axes.labelcolor': TXT,
    'xtick.color': TXT, 'ytick.color': TXT,
    'axes.edgecolor': GRY, 'grid.color': GRY,
    'font.family': 'monospace'
})

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor(BG)
gs_layout = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# — Panel 1: Model Comparison R²
ax1 = fig.add_subplot(gs_layout[0, 0])
names = list(results.keys())
r2vals = [results[m]['R2'] for m in names]
colors = [GRN if v == max(r2vals) else ACC for v in r2vals]
bars = ax1.barh(names, r2vals, color=colors, height=0.6, edgecolor='none')
for bar, val in zip(bars, r2vals):
    ax1.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va='center', color=TXT, fontsize=9)
ax1.set_xlim(0, 1.15)
ax1.invert_yaxis()
ax1.set_title("Model Comparison — R² Score", color=TXT, fontsize=12, pad=10,
              fontweight='bold')
ax1.axvline(0.8, color=ORG, lw=1, ls='--', alpha=0.6, label='0.80 threshold')
ax1.legend(fontsize=8, facecolor=CARD)
ax1.grid(axis='x', alpha=0.3)

# — Panel 2: RMSE Comparison
ax2 = fig.add_subplot(gs_layout[0, 1:])
rmsevals = [results[m]['RMSE'] for m in names]
colorsR = [GRN if v == min(rmsevals) else PRP for v in rmsevals]
ax2.bar(range(len(names)), rmsevals, color=colorsR, edgecolor='none', width=0.6)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels([m.replace(' ', '\n') for m in names], fontsize=8)
ax2.set_title("RMSE (lower = better)", color=TXT, fontsize=11, pad=10,
              fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# — Panel 3: Actual vs Predicted
ax3 = fig.add_subplot(gs_layout[1, 0])
ax3.scatter(y_test.values[:500], best_preds[:500], alpha=0.4, s=12, color=ACC,
            edgecolors='none')
mn, mx = y_test.min(), y_test.max()
ax3.plot([mn, mx], [mn, mx], color=ORG, lw=1.5, ls='--', label='Perfect fit')
ax3.set_xlabel("Actual Price")
ax3.set_ylabel("Predicted Price")
ax3.set_title(f"Actual vs Predicted\nR² = {best_r2:.4f}", color=TXT, fontsize=11,
              fontweight='bold')
ax3.legend(fontsize=8, facecolor=CARD)

# — Panel 4: Residuals
ax4 = fig.add_subplot(gs_layout[1, 1])
residuals = y_test.values - best_preds
ax4.scatter(best_preds[:500], residuals[:500], alpha=0.4, s=12, color=PRP,
            edgecolors='none')
ax4.axhline(0, color=ORG, lw=1.5, ls='--')
ax4.set_xlabel("Predicted Price")
ax4.set_ylabel("Residuals")
ax4.set_title("Residuals Plot", color=TXT, fontsize=11, fontweight='bold')
ax4.grid(alpha=0.2)

# — Panel 5: Residual distribution
ax5 = fig.add_subplot(gs_layout[1, 2])
ax5.hist(residuals, bins=50, color=GRN, edgecolor='none', alpha=0.85)
ax5.axvline(0, color=ORG, lw=1.5, ls='--')
ax5.set_xlabel("Residual")
ax5.set_ylabel("Count")
ax5.set_title("Residual Distribution", color=TXT, fontsize=11, fontweight='bold')

# — Panel 6: Feature Importance
ax6 = fig.add_subplot(gs_layout[2, 0:2])
colors_fi = [ACC if i < 5 else GRY for i in range(len(fi))]
ax6.barh(fi.index, fi.values, color=colors_fi, edgecolor='none', height=0.6)
ax6.invert_yaxis()
ax6.set_title("Feature Importance — Gradient Boosting Tuned", color=TXT,
              fontsize=11, fontweight='bold')
ax6.grid(axis='x', alpha=0.3)

# — Panel 7: Correlation heatmap
ax7 = fig.add_subplot(gs_layout[2, 2])
corr = df[['MedInc', 'AveRooms', 'HouseAge', 'RoomsPerPerson',
           'IncomePerRoom', 'Price']].corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, ax=ax7, cmap=cmap, annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor=BG, cbar=False)
ax7.set_title("Correlation Heatmap", color=TXT, fontsize=11, fontweight='bold')
ax7.tick_params(labelsize=7)

# Title
fig.suptitle("HOUSE PRICE PREDICTION — COMPLETE ML ANALYSIS",
             fontsize=16, color=TXT, fontweight='bold', y=0.98)

plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# =============================================================
# 9. EXPORT EVERYTHING THE WEB DASHBOARD NEEDS
# =============================================================
KEYS = ['alpha', 'max_depth', 'n_estimators', 'learning_rate', 'subsample']
hyper = {nm: {k: v for k, v in m.get_params().items() if k in KEYS} for nm, m in models.items()}
hyper["Gradient Boosting Tuned"] = {k: v for k, v in best.get_params().items() if k in KEYS}

abs_err = np.abs(residuals)
sample = df.sample(200, random_state=42)
sample_pred = best.predict(scaler.transform(X.loc[sample.index]))
sample = sample.round(3).assign(Predicted=np.round(sample_pred, 3))
counts, edges = np.histogram(residuals, bins=30)

payload = {
    "results": results,
    "hyperparameters": hyper,
    "param_grid": param_grid,
    "best_params": gs.best_params_,
    "feature_importance": {k: round(float(v), 5) for k, v in fi.items()},
    "correlation_with_price": df.corr()['Price'].drop('Price').round(4).to_dict(),
    "dataset": {"rows": int(len(df)), "features": int(X.shape[1]), "train": int(len(X_train)),
                "test": int(len(X_test)), "missing_injected": missing_injected,
                "engineered": new_features},
    "stats": df.describe().round(3).T.to_dict('index'),
    "sample": sample.to_dict('records'),
    "scatter": {"actual": np.round(y_test.values[:300], 3).tolist(),
                "predicted": np.round(best_preds[:300], 3).tolist()},
    "residuals": {"counts": counts.tolist(), "edges": np.round(edges, 3).tolist(),
                  "mean": round(float(residuals.mean()), 4), "std": round(float(residuals.std()), 4),
                  "mae": round(float(abs_err.mean()), 4),
                  "median_abs": round(float(np.median(abs_err)), 4),
                  "p90_abs": round(float(np.percentile(abs_err, 90)), 4),
                  "within_50k": round(float((abs_err <= 0.5).mean()), 4)},
}

def _num(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    raise TypeError(type(o))

with open(JSON_PATH, "w") as f:
    json.dump(payload, f, indent=2, default=_num)
with open(JS_PATH, "w") as f:
    f.write("window.DASHBOARD_DATA = " + json.dumps(payload, default=_num) + ";")
df.round(4).to_csv(CSV_PATH, index=False)

print(f"\n[✓] Dashboard saved → {PNG_PATH}")
print(f"[✓] Dashboard data  → {JS_PATH}")
print(f"[✓] Full dataset    → {CSV_PATH}")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)