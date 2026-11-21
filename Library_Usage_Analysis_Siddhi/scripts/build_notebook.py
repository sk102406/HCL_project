"""Builds notebooks/library_usage_analysis.ipynb and executes its code cells
(saving charts + cleaned data). Run from project root."""
import json, io, contextlib, os, sys
os.environ["MPLBACKEND"] = "Agg"

md = lambda s: ("markdown", s.strip("\n"))
code = lambda s: ("code", s.strip("\n"))

cells = [
md("""
# Library Usage Analysis – Data Analyst Project
**Author:** Siddhi Kesarwani | **Admission No:** 2026B01010151 | **Programme:** B.Tech 1st Year, ABES
**Tools:** Python, pandas, NumPy, matplotlib, seaborn, Jupyter

## 1. Business Problem
The college library wants to understand **what drives library usage** so the administration can improve services, raise member engagement and support academic goals.

**Questions we will answer**
1. What is the overall library usage pattern of members?
2. How do visits, stay time and book borrowing differ by member type, department, plan and visit time?
3. Does distance from home affect visits or stay duration?
4. Which members are **low-engagement** (fewer than 4 visits per month)?
5. Is library usage changing month by month?

> The dataset is synthetic (see `scripts/generate_data.py`) and created for learning purposes.
"""),
md("## 2. Import Libraries"),
code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 100
print('Libraries imported successfully!')
"""),
md("## 3. Load and Inspect the Data"),
code("""
df = pd.read_excel('../data/raw/library_usage.xlsx')
print('Rows, Columns:', df.shape)
df.head()
"""),
code("""
df.info()
"""),
code("""
missing = df.isna().sum()
print('Missing values per column:')
print(missing[missing > 0])
print('\\nDuplicate rows:', df.duplicated().sum())
df.describe().round(1)
"""),
code("""
print('Department values   :', df['Department'].unique())
print('Membership_Plan vals:', df['Membership_Plan'].unique())
print('\\nImpossible Age values:')
print(df[(df['Age'] < 0) | (df['Age'] > 100)][['Member_ID', 'Age']])
"""),
md("""
**What we found**
- Missing values in `Preferred_Genre`, `Preferred_Visit_Time`, `Age`, `Avg_Stay_Minutes`, `Distance_From_Home_Km` and `Visit_Date`
- 6 duplicate rows
- Inconsistent text in `Department` (` cse`, `MECHANICAL`, `management `, `  Civil`) and `Membership_Plan` (`premium`, `BASIC `)
- Impossible `Age` values: **-19** and **135**

All of these are fixed in the next step.
"""),
md("## 4. Clean the Data"),
code("""
clean = df.copy()

# 1. Remove duplicate rows
clean = clean.drop_duplicates()

# 2. Standardise text columns (strip spaces, then map to one proper spelling)
dept_map = {'CSE': 'CSE', 'IT': 'IT', 'ECE': 'ECE', 'MECHANICAL': 'Mechanical',
            'CIVIL': 'Civil', 'MANAGEMENT': 'Management'}
clean['Department'] = clean['Department'].str.strip().str.upper().map(dept_map)
clean['Membership_Plan'] = clean['Membership_Plan'].str.strip().str.title()

# 3. Impossible ages (negative or > 100) -> NaN
clean.loc[(clean['Age'] < 0) | (clean['Age'] > 100), 'Age'] = np.nan

# 4. Missing numeric values -> median (robust to outliers)
for col in ['Age', 'Avg_Stay_Minutes', 'Distance_From_Home_Km']:
    clean[col] = clean[col].fillna(clean[col].median())

# 5. Missing categorical values -> most frequent value (mode)
for col in ['Preferred_Genre', 'Preferred_Visit_Time']:
    clean[col] = clean[col].fillna(clean[col].mode()[0])

# 6. Missing Visit_Date -> median date
clean['Visit_Date'] = pd.to_datetime(clean['Visit_Date'])
clean['Visit_Date'] = clean['Visit_Date'].fillna(clean['Visit_Date'].median())

print('Rows after cleaning:', len(clean), '| Missing values left:', clean.isna().sum().sum())
print('Departments:', sorted(clean['Department'].unique()))
print('Plans      :', sorted(clean['Membership_Plan'].unique()))
"""),
md("## 5. Create New Columns (Feature Engineering)"),
code("""
# Engagement level from visits per month  (Low < 4, Moderate 4-8, High >= 9)
def engagement(v):
    if v >= 9: return 'High'
    if v >= 4: return 'Moderate'
    return 'Low'

clean['Engagement_Level'] = clean['Visits_Per_Month'].apply(engagement)

clean['Age_Group'] = pd.cut(clean['Age'], bins=[0, 20, 25, 35, 100],
                            labels=['<21', '21-25', '26-35', '35+'])
clean['Stay_Band'] = pd.cut(clean['Avg_Stay_Minutes'], bins=[0, 30, 60, 90, 200],
                            labels=['<30 min', '30-60 min', '60-90 min', '>90 min'])
clean['Visit_Month'] = clean['Visit_Date'].dt.month_name().str[:3]
clean['Is_Heavy_Borrower'] = (clean['Books_Borrowed_Per_Month'] >= 5).map({True: 'Yes', False: 'No'})

clean[['Member_ID', 'Visits_Per_Month', 'Engagement_Level', 'Age_Group', 'Stay_Band']].head(10)
"""),
code("""
clean.to_csv('../data/processed/library_usage_clean.csv', index=False)
print('Cleaned data saved to data/processed/library_usage_clean.csv')
"""),
md("## 6. Exploratory Data Analysis (EDA)\n### Q1. Overall Library Usage"),
code("""
print('=== Visits Per Month ===');  print(clean['Visits_Per_Month'].describe().round(1))
print('\\n=== Engagement Level ===');  print(clean['Engagement_Level'].value_counts())
print('\\n=== Books Borrowed Per Month ===');  print(clean['Books_Borrowed_Per_Month'].describe().round(1))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Overall Library Usage', fontsize=14, fontweight='bold')
axes[0].hist(clean['Visits_Per_Month'], bins=15, color='#2a6f97', edgecolor='white')
axes[0].set_title('Visits Per Month'); axes[0].set_xlabel('Visits'); axes[0].set_ylabel('Members')
eng = clean['Engagement_Level'].value_counts().reindex(['Low', 'Moderate', 'High'])
axes[1].bar(eng.index, eng.values, color=['#d62828', '#f77f00', '#2a9d8f'])
axes[1].set_title('Engagement Level'); axes[1].set_ylabel('Members')
axes[2].hist(clean['Books_Borrowed_Per_Month'], bins=10, color='#6a4c93', edgecolor='white')
axes[2].set_title('Books Borrowed Per Month'); axes[2].set_xlabel('Books'); axes[2].set_ylabel('Members')
plt.tight_layout(); plt.savefig('../images/01_overall_usage.png'); plt.show()
"""),
md("### Q2. Does Distance Affect Visits and Stay?"),
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Distance and Stay Analysis', fontsize=14, fontweight='bold')
sns.regplot(data=clean, x='Distance_From_Home_Km', y='Visits_Per_Month', ax=axes[0],
            scatter_kws={'alpha': 0.45}, line_kws={'color': 'crimson'})
axes[0].set_title('Distance vs Visits Per Month')
axes[0].set_xlabel('Distance From Home (km)'); axes[0].set_ylabel('Visits Per Month')
sns.regplot(data=clean, x='Avg_Stay_Minutes', y='Books_Borrowed_Per_Month', ax=axes[1],
            scatter_kws={'alpha': 0.45, 'color': 'darkorange'}, line_kws={'color': 'navy'})
axes[1].set_title('Avg Stay vs Books Borrowed')
axes[1].set_xlabel('Avg Stay (minutes)'); axes[1].set_ylabel('Books Borrowed Per Month')
plt.tight_layout(); plt.savefig('../images/02_visits_distance.png'); plt.show()
print('Corr(Distance, Visits) =', round(clean['Distance_From_Home_Km'].corr(clean['Visits_Per_Month']), 2))
print('Corr(Stay, Books)      =', round(clean['Avg_Stay_Minutes'].corr(clean['Books_Borrowed_Per_Month']), 2))
"""),
md("### Q3. Correlation Between Numeric Features"),
code("""
num_cols = ['Age', 'Visits_Per_Month', 'Avg_Stay_Minutes', 'Distance_From_Home_Km', 'Books_Borrowed_Per_Month']
corr = clean[num_cols].corr().round(2)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('../images/03_correlation.png'); plt.show()
corr
"""),
md("### Q4. Group Comparisons (Member Type, Department, Plan, Visit Time)"),
code("""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Group Comparisons', fontsize=14, fontweight='bold')

mt = clean.groupby('Member_Type')['Visits_Per_Month'].mean().sort_values()
axes[0, 0].barh(mt.index, mt.values, color='#2a6f97')
axes[0, 0].set_title('Avg Visits by Member Type'); axes[0, 0].set_xlabel('Avg Visits Per Month')

dep = clean.groupby('Department')['Books_Borrowed_Per_Month'].mean().sort_values()
axes[0, 1].barh(dep.index, dep.values, color='#2a9d8f')
axes[0, 1].set_title('Avg Books Borrowed by Department'); axes[0, 1].set_xlabel('Avg Books Per Month')

plan = clean.groupby('Membership_Plan')[['Avg_Stay_Minutes', 'Books_Borrowed_Per_Month']].mean()
x = np.arange(len(plan)); w = 0.38
axes[1, 0].bar(x - w/2, plan['Avg_Stay_Minutes'] / 10, w, label='Avg Stay (min / 10)', color='#e9c46a')
axes[1, 0].bar(x + w/2, plan['Books_Borrowed_Per_Month'], w, label='Avg Books / Month', color='#e76f51')
axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(plan.index)
axes[1, 0].set_title('Membership Plan: Stay vs Borrowing'); axes[1, 0].legend()

vt = clean['Preferred_Visit_Time'].value_counts().reindex(['Morning', 'Afternoon', 'Evening'])
axes[1, 1].bar(vt.index, vt.values, color=['#8ecae6', '#219ebc', '#023047'])
axes[1, 1].set_title('Preferred Visit Time'); axes[1, 1].set_ylabel('Members')
plt.tight_layout(); plt.savefig('../images/04_group_comparison.png'); plt.show()

clean.groupby('Member_Type')[['Visits_Per_Month', 'Avg_Stay_Minutes', 'Books_Borrowed_Per_Month']].mean().round(1)
"""),
md("### Q5. Member Demographics"),
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Member Demographics', fontsize=14, fontweight='bold')
clean['Member_Type'].value_counts().plot.pie(autopct='%1.0f%%', ax=axes[0], startangle=90)
axes[0].set_ylabel(''); axes[0].set_title('Member Type')
clean['Department'].value_counts().plot.bar(ax=axes[1], color='#2a6f97')
axes[1].set_title('Department'); axes[1].tick_params(axis='x', rotation=30)
clean['Preferred_Genre'].value_counts().plot.bar(ax=axes[2], color='#6a4c93')
axes[2].set_title('Preferred Genre'); axes[2].tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig('../images/05_member_demographics.png'); plt.show()
"""),
md("### Q6. Spread of Stay Time by Member Type"),
code("""
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=clean, x='Member_Type', y='Avg_Stay_Minutes', ax=ax)
ax.set_title('Average Stay by Member Type', fontsize=14, fontweight='bold')
ax.set_xlabel(''); ax.set_ylabel('Avg Stay (minutes)')
plt.tight_layout(); plt.savefig('../images/06_boxplot_member_type.png'); plt.show()
"""),
md("### Q7. Monthly Trend"),
code("""
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
trend = (clean.groupby('Visit_Month')['Visits_Per_Month'].mean().reindex(month_order))
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(trend.index, trend.values, marker='o', color='#d62828')
ax.set_title('Average Visits Per Month by Last-Visit Month', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Visits Per Month'); ax.set_xlabel('Month (2026)')
plt.tight_layout(); plt.savefig('../images/07_monthly_trend.png'); plt.show()
trend.round(1)
"""),
md("### Q8. Low-Engagement Members (At-Risk)"),
code("""
low = clean[clean['Engagement_Level'] == 'Low']
print(f'Low-engagement members (< 4 visits/month): {len(low)}')
print(f'Share of all members: {len(low) / len(clean) * 100:.1f}%\\n')
print('By Member Type:'); print(low['Member_Type'].value_counts()); print()
print('By Department:'); print(low['Department'].value_counts()); print()
print('Average distance  low vs others (km):',
      round(low['Distance_From_Home_Km'].mean(), 1), 'vs',
      round(clean[clean['Engagement_Level'] != 'Low']['Distance_From_Home_Km'].mean(), 1))

ct = pd.crosstab(clean['Department'], clean['Engagement_Level'], normalize='index')[['Low', 'Moderate', 'High']] * 100
ct.plot(kind='bar', stacked=True, figsize=(9, 5), color=['#d62828', '#f77f00', '#2a9d8f'])
plt.title('Engagement Mix by Department (%)', fontsize=14, fontweight='bold')
plt.ylabel('% of members'); plt.xlabel(''); plt.xticks(rotation=25); plt.legend(title='Level')
plt.tight_layout(); plt.savefig('../images/08_engagement_by_dept.png'); plt.show()
low[['Member_ID', 'Member_Type', 'Department', 'Visits_Per_Month', 'Distance_From_Home_Km']].head(10)
"""),
md("### Q9. Engagement by Plan and Distance Band"),
code("""
clean['Distance_Band'] = pd.cut(clean['Distance_From_Home_Km'], bins=[0, 4, 8, 12, 20],
                                labels=['0-4 km', '4-8 km', '8-12 km', '12+ km'])
pivot = clean.pivot_table(index='Distance_Band', columns='Membership_Plan',
                          values='Visits_Per_Month', aggfunc='mean', observed=True).round(1)
ax = pivot.plot(kind='bar', figsize=(8, 5), color=['#8d99ae', '#ef233c'])
ax.set_title('Avg Visits: Distance Band x Plan', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Visits Per Month'); ax.set_xlabel(''); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig('../images/09_distance_band_plan.png'); plt.show()
pivot
"""),
md("""
## 7. Key Insights and Recommendations
*(Numbers below are produced by the cells above – see the README for the exact values.)*

### Insights
1. **Overall usage:** members visit about 8 times a month on average; roughly 10% are Low engagement (< 4 visits).
2. **Distance effect:** visits fall steadily as distance from home increases (strong negative correlation).
3. **Stay vs borrowing:** members who stay longer borrow more books.
4. **Member type:** Research Scholars and Faculty stay longer than Students.
5. **Plan:** Premium members visit and borrow more than Basic members.
6. **Timing:** afternoon is the busiest slot, followed by morning.
7. **Genre:** Technical and Competitive Exams are the most preferred genres.

### Recommendations
- **Target low-engagement members** with reminders and personal outreach (email / SMS).
- **Reduce the distance barrier** with a shuttle or a satellite reading corner for far-away members.
- **Promote the Premium plan** by showing the extra borrowing benefits.
- **Staff for the afternoon peak** and add quiet-study seating in the evening.
- **Genre-based events** such as Competitive-Exam prep sessions to attract low-engagement members.
- **Track monthly** to catch drops in engagement early.
"""),
]

# ----- write notebook JSON (no outputs, learner re-runs it) --------------------
nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, (kind, src) in enumerate(cells):
    lines = src.split("\n"); lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "id": f"c{i:03d}", "metadata": {}, "source": lines}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
with open("notebooks/library_usage_analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

# ----- execute code cells here (cwd = notebooks) to create charts/CSV ----------
os.chdir("notebooks")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")
ns = {}
for kind, src in cells:
    if kind != "code": continue
    # emulate notebook: print last bare expression
    lines = src.split("\n")
    last = lines[-1]
    try:
        compile(last, "<l>", "eval"); is_expr = not last.startswith((" ", "\t")) and "=" not in last.split("(")[0]
    except SyntaxError:
        is_expr = False
    if is_expr:
        exec("\n".join(lines[:-1]), ns)
        print(eval(last, ns))
    else:
        exec(src, ns)
    print("-" * 50)
