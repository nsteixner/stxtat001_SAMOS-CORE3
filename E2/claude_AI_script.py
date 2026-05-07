import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD & PARSE ──────────────────────────────────────────────────────────
df = pd.read_csv(
    '/mnt/user-data/uploads/SAA2_WC_2017_metocean_10min_avg.csv',
    na_values=['', 'NaN', 'nan', 'NA', '-9999', '-999'],
    parse_dates=['TIME_SERVER'],
    index_col='TIME_SERVER'
)

print("=== DATA LOADED ===")
print(f"Shape: {df.shape}")
print(f"Index dtype: {df.index.dtype}")
print(f"Date range: {df.index[0]} → {df.index[-1]}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")

# Convert NMEA latitude (DDMM.mmm) to decimal degrees, apply sign for S
def nmea_to_decimal(lat_series, ns_series):
    deg = (lat_series // 100).astype(float)
    minutes = lat_series % 100
    decimal = deg + minutes / 60
    decimal[ns_series == 'S'] *= -1
    return decimal

df['LAT_DD'] = nmea_to_decimal(df['LATITUDE'], df['N_S'])

# ── 2. SELECT DEPARTURE → JULY 4 INCLUSIVE ───────────────────────────────────
cutoff = '2017-07-04'
sel = df.loc[:cutoff].copy()

print(f"\n=== SELECTED DATA ===")
print(f"Rows: {len(sel)}")
print(f"Period: {sel.index[0]} → {sel.index[-1]}")

# ── 3. TIME SERIES OF SST (TSG_TEMP) ─────────────────────────────────────────
with plt.style.context('grayscale'):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sel.index, sel['TSG_TEMP'], linewidth=0.8, label='SST')
    ax.set_xlabel('Date (UTC)', fontsize=11)
    ax.set_ylabel('Sea Surface Temperature (°C)', fontsize=11)
    ax.set_title('Sea Surface Temperature – Southern Ocean Cruise 2017\n(Departure → 4 July 2017)',
                 fontsize=12)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d %b'))
    fig.autofmt_xdate()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/fig1_temperature_timeseries.png', dpi=150)
    plt.close()
    print("\nFig 1 saved (grayscale) → fig1_temperature_timeseries.png")

# ── 4. SALINITY HISTOGRAM ─────────────────────────────────────────────────────
bins = np.arange(30, 35 + 0.5, 0.5)   # 30, 30.5, … 35

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(sel['TSG_SALINITY'].dropna(), bins=bins, color='steelblue',
        edgecolor='white', linewidth=0.6)
ax.set_xlabel('Salinity (PSU)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Salinity Distribution – Southern Ocean Cruise 2017', fontsize=12)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig2_salinity_histogram.png', dpi=150)
plt.close()
print("Fig 2 saved → fig2_salinity_histogram.png")

# ── 5. STATISTICS TABLE ───────────────────────────────────────────────────────
def iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)

stats = pd.DataFrame({
    'Variable': ['Sea Surface Temperature (°C)', 'Salinity (PSU)'],
    'Mean':  [sel['TSG_TEMP'].mean(),        sel['TSG_SALINITY'].mean()],
    'Std Dev': [sel['TSG_TEMP'].std(),       sel['TSG_SALINITY'].std()],
    'IQR':   [iqr(sel['TSG_TEMP']),          iqr(sel['TSG_SALINITY'])]
})
stats = stats.set_index('Variable').round(4)
print("\n=== STATISTICS TABLE ===")
print(stats.to_string())

# Save stats as a nicely rendered figure/table
fig, ax = plt.subplots(figsize=(7, 2))
ax.axis('off')
tbl = ax.table(
    cellText=stats.reset_index().values,
    colLabels=['Variable', 'Mean', 'Std Dev', 'IQR'],
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
# Style header row
for j in range(4):
    tbl[0, j].set_facecolor('#2c5f8a')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
plt.title('Descriptive Statistics – Temperature & Salinity', fontsize=11,
          pad=10, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig3_statistics_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved → fig3_statistics_table.png")

# ── 6. SCATTER: WIND SPEED vs AIR TEMP, COLOURED BY LATITUDE ──────────────────
scatter_df = sel[['WIND_SPEED_TRUE', 'AIR_TEMPERATURE', 'LAT_DD']].dropna()

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    scatter_df['WIND_SPEED_TRUE'],
    scatter_df['AIR_TEMPERATURE'],
    c=scatter_df['LAT_DD'],
    cmap='plasma_r',         # darker = more southerly (lower lat)
    alpha=0.6,
    s=18,
    linewidths=0
)
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Latitude (°S, negative)', fontsize=10)
ax.set_xlabel('True Wind Speed (m s⁻¹)', fontsize=11)
ax.set_ylabel('Air Temperature (°C)', fontsize=11)
ax.set_title('Wind Speed vs Air Temperature\n(colour = latitude)', fontsize=12)
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fig4_windspeed_airtemp_scatter.png', dpi=300)
plt.close()
print("Fig 4 saved (300 DPI PNG) → fig4_windspeed_airtemp_scatter.png")

print("\n✓ All outputs written to /mnt/user-data/outputs/")
