
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. STYLE SETUP
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d2e',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
    'text.color': '#e0e0e0',
    'axes.titlecolor': '#ffffff',
    'grid.color': '#2a2d3e',
})

PALETTE = ['#6c63ff', '#ff6584', '#43e97b', '#f7971e', '#4facfe', '#f093fb']
BG      = '#0f1117'
CARD    = '#1a1d2e'
ACCENT  = '#6c63ff'

# ─────────────────────────────────────────────
# 1. GENERATE RAW (DIRTY) DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
N = 1200

categories = ['Electronics', 'Clothing', 'Groceries', 'Home & Living', 'Sports', 'Books']
regions    = ['North', 'South', 'East', 'West', 'Central']
channels   = ['Online', 'In-Store', 'Mobile App']

base_dates = pd.date_range('2023-01-01', '2023-12-31', periods=N)

raw = pd.DataFrame({
    'order_id':    [f'ORD-{i:05d}' for i in range(N)],
    'customer_id': np.random.randint(1000, 9999, N),
    'date':        base_dates,
    'category':    np.random.choice(categories, N, p=[0.25,0.20,0.18,0.15,0.12,0.10]),
    'region':      np.random.choice(regions, N),
    'channel':     np.random.choice(channels, N, p=[0.45,0.35,0.20]),
    'quantity':    np.random.randint(1, 15, N).astype(float),
    'unit_price':  np.round(np.random.uniform(5, 500, N), 2),
    'discount':    np.round(np.random.uniform(0, 0.4, N), 2),
    'rating':      np.round(np.random.uniform(1, 5, N), 1),
})
raw['revenue'] = np.round(raw['quantity'] * raw['unit_price'] * (1 - raw['discount']), 2)

# --- Inject Dirt ---
# Missing values
for col, rate in [('rating', 0.08), ('discount', 0.05), ('region', 0.04), ('channel', 0.03)]:
    idx = np.random.choice(N, int(N * rate), replace=False)
    raw.loc[idx, col] = np.nan

# Duplicates
dup_idx = np.random.choice(N, 40, replace=False)
raw = pd.concat([raw, raw.iloc[dup_idx]], ignore_index=True)

# Outliers
outlier_idx = np.random.choice(len(raw), 25, replace=False)
raw.loc[outlier_idx, 'unit_price'] = np.random.uniform(5000, 12000, 25)
raw.loc[outlier_idx[:10], 'quantity'] = np.random.uniform(-5, 0, 10)

# Inconsistent categories
noise_idx = np.random.choice(len(raw), 30, replace=False)
raw.loc[noise_idx, 'category'] = np.random.choice(
    ['electronics', 'CLOTHING', 'Groceries ', ' Books', 'home & living'], 30)

raw_shape = raw.shape
print(f"✅ Raw dataset created: {raw_shape[0]} rows × {raw_shape[1]} cols")
print(f"   Missing values:\n{raw.isnull().sum()[raw.isnull().sum()>0]}")
print(f"   Duplicates: {raw.duplicated(subset='order_id').sum()}")


# ─────────────────────────────────────────────
# 2. DATA CLEANING PIPELINE
# ─────────────────────────────────────────────
print("\n🔧 Running cleaning pipeline...")
df = raw.copy()

# Step 1: Remove duplicates
before_dup = len(df)
df.drop_duplicates(subset='order_id', inplace=True)
removed_dups = before_dup - len(df)

# Step 2: Standardize category names
df['category'] = (df['category']
                  .str.strip()
                  .str.title()
                  .replace({'Home & Living': 'Home & Living',
                            'Home &Amp; Living': 'Home & Living'}))
valid_cats = ['Electronics', 'Clothing', 'Groceries', 'Home & Living', 'Sports', 'Books']
df = df[df['category'].isin(valid_cats)]

# Step 3: Fix negative quantities → absolute value
df['quantity'] = df['quantity'].abs()

# Step 4: Remove price outliers (IQR method)
Q1, Q3 = df['unit_price'].quantile(0.25), df['unit_price'].quantile(0.75)
IQR = Q3 - Q1
before_out = len(df)
df = df[df['unit_price'].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)]
removed_out = before_out - len(df)

# Step 5: Fill missing values
df['rating'].fillna(df['rating'].median(), inplace=True)
df['discount'].fillna(df['discount'].median(), inplace=True)
df['region'].fillna(df['region'].mode()[0], inplace=True)
df['channel'].fillna(df['channel'].mode()[0], inplace=True)

# Step 6: Recalculate revenue with clean data
df['revenue'] = np.round(df['quantity'] * df['unit_price'] * (1 - df['discount']), 2)

# Step 7: Add derived columns
df['date'] = pd.to_datetime(df['date'])
df['month']    = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')
df['quarter']  = df['date'].dt.quarter
df['weekday']  = df['date'].dt.day_name()
df['profit']   = np.round(df['revenue'] * np.random.uniform(0.15, 0.40, len(df)), 2)

print(f"   Duplicates removed : {removed_dups}")
print(f"   Outliers removed   : {removed_out}")
print(f"   Missing values filled")
print(f"✅ Clean dataset: {df.shape[0]} rows × {df.shape[1]} cols\n")


# ─────────────────────────────────────────────
# 3. SUMMARY STATS
# ─────────────────────────────────────────────
total_revenue = df['revenue'].sum()
total_orders  = len(df)
avg_order_val = df['revenue'].mean()
avg_rating    = df['rating'].mean()

monthly_rev   = df.groupby('month')['revenue'].sum()
cat_rev       = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
region_rev    = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
channel_rev   = df.groupby('channel')['revenue'].sum()
weekday_ord   = df.groupby('weekday')['order_id'].count().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
cat_rating    = df.groupby('category')['rating'].mean().sort_values()
quarter_rev   = df.groupby('quarter')['revenue'].sum()


# ─────────────────────────────────────────────
# 4. DASHBOARD — PAGE 1: CLEANING REPORT
# ─────────────────────────────────────────────
fig1 = plt.figure(figsize=(18, 10), facecolor=BG)
fig1.suptitle('DATA CLEANING REPORT', fontsize=22, fontweight='bold',
              color='white', y=0.97, fontfamily='monospace')

gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97, top=0.90, bottom=0.08)

# --- KPI Cards (top row, manually placed) ---
kpi_data = [
    ("Raw Records",    f"{raw_shape[0]:,}",    '#6c63ff'),
    ("Duplicates Removed", f"{removed_dups}",  '#ff6584'),
    ("Outliers Removed",   f"{removed_out}",   '#f7971e'),
    ("Clean Records",  f"{df.shape[0]:,}",     '#43e97b'),
    ("Missing Filled", "100%",                 '#4facfe'),
    ("Data Quality",   f"{df.shape[0]/raw_shape[0]*100:.1f}%", '#f093fb'),
]

for i, (label, value, color) in enumerate(kpi_data):
    ax = fig1.add_subplot(gs[0, i % 3]) if i < 3 else None
    if i < 3:
        ax.set_facecolor(CARD)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.65, value, ha='center', va='center',
                fontsize=28, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha='center', va='center',
                fontsize=10, color='#a0a0a0', transform=ax.transAxes)
        ax.set_title('', pad=0)

# Bottom row: missing values bar, outlier box, cleaning steps text
ax_miss = fig1.add_subplot(gs[1, 0])
ax_miss.set_facecolor(CARD)
miss_before = raw.isnull().sum().sort_values(ascending=False)
miss_before = miss_before[miss_before > 0]
bars = ax_miss.barh(miss_before.index, miss_before.values, color=PALETTE[:len(miss_before)], height=0.5)
ax_miss.set_title('Missing Values (Before)', color='white', fontsize=12, pad=8)
ax_miss.set_xlabel('Count', color='#a0a0a0')
for bar, val in zip(bars, miss_before.values):
    ax_miss.text(val + 1, bar.get_y() + bar.get_height()/2, str(val),
                 va='center', color='white', fontsize=9)

ax_out = fig1.add_subplot(gs[1, 1])
ax_out.set_facecolor(CARD)
ax_out.boxplot([raw['unit_price'].dropna(), df['unit_price']],
               labels=['Raw', 'Cleaned'],
               patch_artist=True,
               boxprops=dict(facecolor=PALETTE[0], alpha=0.7),
               medianprops=dict(color='#f7971e', linewidth=2),
               whiskerprops=dict(color='#a0a0a0'),
               capprops=dict(color='#a0a0a0'),
               flierprops=dict(marker='o', color='#ff6584', alpha=0.5, markersize=4))
ax_out.set_title('Outlier Treatment — Unit Price', color='white', fontsize=12, pad=8)
ax_out.set_ylabel('Unit Price (₹)', color='#a0a0a0')

ax_steps = fig1.add_subplot(gs[1, 2])
ax_steps.set_facecolor(CARD)
ax_steps.axis('off')
ax_steps.set_title('Cleaning Pipeline Steps', color='white', fontsize=12, pad=8)
steps = [
    ("1", "Remove Duplicate Order IDs",      '#6c63ff'),
    ("2", "Standardize Category Names",       '#ff6584'),
    ("3", "Fix Negative Quantities (abs)",    '#43e97b'),
    ("4", "IQR Outlier Removal — Price",      '#f7971e'),
    ("5", "Fill Missing: Median / Mode",      '#4facfe'),
    ("6", "Recalculate Revenue Column",       '#f093fb'),
    ("7", "Add Derived Time Features",        '#6c63ff'),
]
for j, (num, step, clr) in enumerate(steps):
    y = 0.88 - j * 0.13
    ax_steps.add_patch(mpatches.FancyBboxPatch((0.02, y-0.05), 0.96, 0.10,
        boxstyle="round,pad=0.01", facecolor=clr+'22', edgecolor=clr, linewidth=1,
        transform=ax_steps.transAxes))
    ax_steps.text(0.08, y, f"Step {num}", transform=ax_steps.transAxes,
                  fontsize=8, color=clr, fontweight='bold', va='center')
    ax_steps.text(0.28, y, step, transform=ax_steps.transAxes,
                  fontsize=8, color='white', va='center')

# Add secondary KPIs as text overlay on top right
for i, (label, value, color) in enumerate(kpi_data[3:]):
    xi = i % 3
    ax2 = fig1.add_axes([0.06 + xi*0.308, 0.62, 0.27, 0.12])
    ax2.set_facecolor(CARD)
    for spine in ax2.spines.values(): spine.set_visible(False)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(0.5, 0.65, value, ha='center', va='center',
             fontsize=28, fontweight='bold', color=color, transform=ax2.transAxes)
    ax2.text(0.5, 0.18, label, ha='center', va='center',
             fontsize=10, color='#a0a0a0', transform=ax2.transAxes)

plt.savefig('/mnt/user-data/outputs/01_cleaning_report.png', dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.close()
print("✅ Saved: 01_cleaning_report.png")


# ─────────────────────────────────────────────
# 5. DASHBOARD — PAGE 2: REVENUE INSIGHTS
# ─────────────────────────────────────────────
fig2 = plt.figure(figsize=(18, 12), facecolor=BG)
fig2.suptitle('REVENUE & SALES INSIGHTS DASHBOARD', fontsize=22, fontweight='bold',
              color='white', y=0.98, fontfamily='monospace')

gs2 = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.50, wspace=0.35,
                        left=0.06, right=0.97, top=0.92, bottom=0.06)

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# --- 1. Monthly Revenue Line ---
ax1 = fig2.add_subplot(gs2[0, :2])
ax1.set_facecolor(CARD)
x = range(1, 13)
y = [monthly_rev.get(m, 0) for m in x]
ax1.fill_between(x, y, alpha=0.15, color=ACCENT)
ax1.plot(x, y, color=ACCENT, linewidth=2.5, marker='o', markersize=6, markerfacecolor='white')
ax1.set_xticks(x); ax1.set_xticklabels(months, fontsize=9)
ax1.set_title('Monthly Revenue Trend (2023)', color='white', fontsize=13, pad=10)
ax1.set_ylabel('Revenue (₹)', color='#a0a0a0')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v/1000:.0f}K'))
# Annotate peak
peak_m = np.argmax(y)
ax1.annotate(f'Peak\n₹{y[peak_m]/1000:.0f}K', xy=(peak_m+1, y[peak_m]),
             xytext=(peak_m+1, y[peak_m]*1.08),
             ha='center', color='#43e97b', fontsize=8, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#43e97b', lw=1.2))

# --- 2. Category Revenue Donut ---
ax2 = fig2.add_subplot(gs2[0, 2])
ax2.set_facecolor(CARD)
wedges, texts, autotexts = ax2.pie(
    cat_rev.values, labels=cat_rev.index, autopct='%1.1f%%',
    colors=PALETTE, startangle=140,
    pctdistance=0.75, wedgeprops=dict(width=0.55, edgecolor=CARD, linewidth=2))
for t in autotexts: t.set_color('white'); t.set_fontsize(8)
for t in texts: t.set_color('#a0a0a0'); t.set_fontsize(8)
ax2.set_title('Revenue by Category', color='white', fontsize=13, pad=10)

# --- 3. Region Bar Chart ---
ax3 = fig2.add_subplot(gs2[1, 0])
ax3.set_facecolor(CARD)
colors3 = [PALETTE[i] for i in range(len(region_rev))]
bars3 = ax3.bar(region_rev.index, region_rev.values, color=colors3, width=0.6, edgecolor=CARD)
ax3.set_title('Revenue by Region', color='white', fontsize=12, pad=8)
ax3.set_ylabel('Revenue (₹)', color='#a0a0a0')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v/1000:.0f}K'))
for bar in bars3:
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
             f'₹{bar.get_height()/1000:.0f}K', ha='center', va='bottom',
             color='white', fontsize=8, fontweight='bold')

# --- 4. Channel Stacked Bar ---
ax4 = fig2.add_subplot(gs2[1, 1])
ax4.set_facecolor(CARD)
ch_cat = df.groupby(['channel','category'])['revenue'].sum().unstack(fill_value=0)
bot = np.zeros(len(ch_cat))
for i, cat in enumerate(ch_cat.columns):
    ax4.bar(ch_cat.index, ch_cat[cat], bottom=bot, label=cat,
            color=PALETTE[i], edgecolor=CARD, linewidth=0.5)
    bot += ch_cat[cat].values
ax4.set_title('Revenue by Channel & Category', color='white', fontsize=12, pad=8)
ax4.set_ylabel('Revenue (₹)', color='#a0a0a0')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v/1000:.0f}K'))
ax4.legend(fontsize=7, loc='upper right', framealpha=0.2, labelcolor='white')

# --- 5. Weekday Orders Heatmap-style bar ---
ax5 = fig2.add_subplot(gs2[1, 2])
ax5.set_facecolor(CARD)
wday_vals = weekday_ord.values
wday_colors = [PALETTE[0] if v == max(wday_vals) else '#3a3d5e' for v in wday_vals]
wday_colors[list(wday_vals).index(max(wday_vals))] = '#43e97b'
ax5.bar(range(7), wday_vals, color=wday_colors, width=0.6)
ax5.set_xticks(range(7))
ax5.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], fontsize=8)
ax5.set_title('Orders by Weekday', color='white', fontsize=12, pad=8)
ax5.set_ylabel('Orders', color='#a0a0a0')

# --- 6. Rating Distribution ---
ax6 = fig2.add_subplot(gs2[2, 0])
ax6.set_facecolor(CARD)
ax6.hist(df['rating'], bins=20, color=PALETTE[4], edgecolor=CARD, alpha=0.85)
ax6.axvline(df['rating'].mean(), color='#ff6584', linewidth=2, linestyle='--',
            label=f'Mean: {df["rating"].mean():.2f}')
ax6.legend(fontsize=9, framealpha=0.2, labelcolor='white')
ax6.set_title('Customer Rating Distribution', color='white', fontsize=12, pad=8)
ax6.set_xlabel('Rating', color='#a0a0a0')
ax6.set_ylabel('Frequency', color='#a0a0a0')

# --- 7. Category Avg Rating ---
ax7 = fig2.add_subplot(gs2[2, 1])
ax7.set_facecolor(CARD)
colors7 = [PALETTE[i] for i in range(len(cat_rating))]
hbars = ax7.barh(cat_rating.index, cat_rating.values, color=colors7, height=0.55)
ax7.set_xlim(0, 5.5)
ax7.axvline(3, color='#ff6584', linewidth=1, linestyle='--', alpha=0.6)
ax7.set_title('Avg Rating by Category', color='white', fontsize=12, pad=8)
ax7.set_xlabel('Average Rating', color='#a0a0a0')
for bar, val in zip(hbars, cat_rating.values):
    ax7.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', color='white', fontsize=9)

# --- 8. Quarterly Revenue ---
ax8 = fig2.add_subplot(gs2[2, 2])
ax8.set_facecolor(CARD)
q_colors = [PALETTE[i] for i in range(4)]
bars8 = ax8.bar(['Q1','Q2','Q3','Q4'], quarter_rev.values, color=q_colors, width=0.55)
ax8.set_title('Quarterly Revenue', color='white', fontsize=12, pad=8)
ax8.set_ylabel('Revenue (₹)', color='#a0a0a0')
ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v/1000:.0f}K'))
for bar in bars8:
    ax8.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
             f'₹{bar.get_height()/1000:.0f}K', ha='center', va='bottom',
             color='white', fontsize=9, fontweight='bold')

plt.savefig('/mnt/user-data/outputs/02_revenue_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Saved: 02_revenue_dashboard.png")


# ─────────────────────────────────────────────
# 6. DASHBOARD — PAGE 3: ADVANCED ANALYTICS
# ─────────────────────────────────────────────
fig3 = plt.figure(figsize=(18, 12), facecolor=BG)
fig3.suptitle('ADVANCED ANALYTICS & STORYTELLING', fontsize=22, fontweight='bold',
              color='white', y=0.98, fontfamily='monospace')

gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.35,
                        left=0.06, right=0.97, top=0.92, bottom=0.06)

# --- 1. Revenue vs Discount Scatter ---
ax_s1 = fig3.add_subplot(gs3[0, 0])
ax_s1.set_facecolor(CARD)
cat_map = {c: i for i, c in enumerate(categories)}
colors_s = [PALETTE[cat_map.get(c, 0)] for c in df['category']]
sc = ax_s1.scatter(df['discount']*100, df['revenue'], c=colors_s,
                   alpha=0.4, s=18, edgecolors='none')
ax_s1.set_title('Revenue vs Discount %', color='white', fontsize=12, pad=8)
ax_s1.set_xlabel('Discount (%)', color='#a0a0a0')
ax_s1.set_ylabel('Revenue (₹)', color='#a0a0a0')
patches = [mpatches.Patch(color=PALETTE[i], label=c) for i, c in enumerate(categories)]
ax_s1.legend(handles=patches, fontsize=7, framealpha=0.2, labelcolor='white')

# --- 2. Correlation Heatmap ---
ax_s2 = fig3.add_subplot(gs3[0, 1])
ax_s2.set_facecolor(CARD)
num_cols = ['quantity', 'unit_price', 'discount', 'revenue', 'rating', 'profit']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, ax=ax_s2, cmap=cmap, center=0, annot=True,
            fmt='.2f', annot_kws={'size': 8}, linewidths=0.5,
            linecolor=BG, cbar_kws={'shrink': 0.8})
ax_s2.set_title('Feature Correlation Matrix', color='white', fontsize=12, pad=8)
ax_s2.tick_params(colors='#a0a0a0', labelsize=8)

# --- 3. Revenue CDF ---
ax_s3 = fig3.add_subplot(gs3[0, 2])
ax_s3.set_facecolor(CARD)
sorted_rev = np.sort(df['revenue'])
cdf = np.arange(1, len(sorted_rev)+1) / len(sorted_rev)
ax_s3.plot(sorted_rev, cdf, color=PALETTE[4], linewidth=2)
ax_s3.fill_betweenx(cdf, sorted_rev, alpha=0.1, color=PALETTE[4])
p80 = np.percentile(sorted_rev, 80)
ax_s3.axvline(p80, color='#ff6584', linestyle='--', linewidth=1.5, label=f'80th pct: ₹{p80:.0f}')
ax_s3.legend(fontsize=9, framealpha=0.2, labelcolor='white')
ax_s3.set_title('Revenue CDF (Cumulative Distribution)', color='white', fontsize=12, pad=8)
ax_s3.set_xlabel('Revenue (₹)', color='#a0a0a0')
ax_s3.set_ylabel('Cumulative Probability', color='#a0a0a0')

# --- 4. Monthly Revenue by Category (Heatmap) ---
ax_s4 = fig3.add_subplot(gs3[1, :2])
ax_s4.set_facecolor(CARD)
pivot = df.groupby(['month_name', 'category'])['revenue'].sum().unstack(fill_value=0)
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = pivot.reindex([m for m in month_order if m in pivot.index])
sns.heatmap(pivot.T, ax=ax_s4, cmap='RdYlGn', fmt='.0f',
            linewidths=0.5, linecolor=BG,
            cbar_kws={'shrink': 0.8, 'label': 'Revenue (₹)'})
ax_s4.set_title('Monthly Revenue Heatmap by Category', color='white', fontsize=12, pad=8)
ax_s4.set_xlabel('Month', color='#a0a0a0')
ax_s4.set_ylabel('Category', color='#a0a0a0')
ax_s4.tick_params(colors='#a0a0a0', labelsize=8)

# --- 5. Key Insights Text Box ---
ax_s5 = fig3.add_subplot(gs3[1, 2])
ax_s5.set_facecolor(CARD)
ax_s5.axis('off')
ax_s5.set_title('📊 Key Story Insights', color='white', fontsize=12, pad=8)

top_cat = cat_rev.idxmax()
top_region = region_rev.idxmax()
top_channel = channel_rev.idxmax()
best_day = weekday_ord.idxmax()
top_q = f"Q{quarter_rev.idxmax()}"

insights = [
    (f"Total Revenue: ₹{total_revenue:,.0f}", '#43e97b'),
    (f"Total Orders: {total_orders:,}", '#4facfe'),
    (f"Avg Order Value: ₹{avg_order_val:.0f}", '#6c63ff'),
    (f"Top Category: {top_cat}", '#f7971e'),
    (f"Top Region: {top_region}", '#ff6584'),
    (f"Best Channel: {top_channel}", '#f093fb'),
    (f"Busiest Day: {best_day}", '#43e97b'),
    (f"Best Quarter: {top_q}", '#4facfe'),
    (f"Avg Rating: {avg_rating:.2f} / 5.0", '#6c63ff'),
]
for j, (txt, clr) in enumerate(insights):
    y = 0.90 - j * 0.10
    ax_s5.add_patch(mpatches.FancyBboxPatch((0.02, y-0.04), 0.96, 0.08,
        boxstyle="round,pad=0.01", facecolor=clr+'18', edgecolor=clr+'55', linewidth=1,
        transform=ax_s5.transAxes))
    ax_s5.text(0.08, y, '●', transform=ax_s5.transAxes, fontsize=10,
               color=clr, va='center')
    ax_s5.text(0.16, y, txt, transform=ax_s5.transAxes, fontsize=9,
               color='white', va='center')

plt.savefig('/mnt/user-data/outputs/03_advanced_analytics.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Saved: 03_advanced_analytics.png")


# ─────────────────────────────────────────────
# 7. EXPORT CLEAN CSV
# ─────────────────────────────────────────────
df.to_csv('/mnt/user-data/outputs/cleaned_retail_data.csv', index=False)
print("✅ Saved: cleaned_retail_data.csv")

print("\n" + "="*55)
print("  PROJECT COMPLETE — All outputs saved!")
print("="*55)
print(f"  Revenue         : ₹{total_revenue:>12,.0f}")
print(f"  Orders          : {total_orders:>12,}")
print(f"  Avg Order Value : ₹{avg_order_val:>12,.0f}")
print(f"  Avg Rating      : {avg_rating:>12.2f}")
print(f"  Top Category    : {top_cat:>15}")
print(f"  Top Region      : {top_region:>15}")
print("="*55)
