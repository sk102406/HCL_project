"""Generate the raw Library Usage dataset used in this project.

Author : Siddhi Kesarwani (Admission No: 2026B01010151)
Output : data/raw/library_usage.csv and data/raw/library_usage.csv

The data is SYNTHETIC. It is built so that realistic relationships exist
(distance lowers visits, longer stays raise borrowing, ...) and so that
common data-quality problems are present for cleaning practice.
Run from the project root:  python scripts/generate_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N = 250  # genuine records (6 duplicates are appended later -> 256 rows)

member_type = rng.choice(["Student", "Research Scholar", "Staff", "Faculty"],
                         N, p=[0.60, 0.15, 0.15, 0.10])
department = rng.choice(["CSE", "IT", "ECE", "Mechanical", "Civil", "Management"],
                        N, p=[0.25, 0.15, 0.20, 0.15, 0.10, 0.15])
genre = rng.choice(["Fiction", "Technical", "Self-Help", "Competitive Exams",
                    "Reference", "Biography"], N, p=[0.18, 0.28, 0.12, 0.22, 0.12, 0.08])
plan = rng.choice(["Basic", "Premium"], N, p=[0.60, 0.40])
visit_time = rng.choice(["Morning", "Afternoon", "Evening"], N, p=[0.30, 0.45, 0.25])

age = np.where(member_type == "Student", rng.integers(17, 25, N),
      np.where(member_type == "Research Scholar", rng.integers(23, 32, N),
      np.where(member_type == "Staff", rng.integers(25, 56, N),
               rng.integers(30, 61, N)))).astype(float)

distance = np.round(rng.uniform(0.4, 16.0, N), 1)

# Visits: fewer when far away, more with Premium / Research Scholars
visits = (10.5 - 0.42 * distance + 1.8 * (plan == "Premium")
          + 1.5 * (member_type == "Research Scholar") + rng.normal(0, 2.2, N))
visits = np.clip(np.round(visits), 1, 18).astype(int)

# Stay: longer for Research Scholars and Faculty
stay = (38 + 22 * np.isin(member_type, ["Research Scholar", "Faculty"])
        + rng.normal(0, 17, N))
stay = np.clip(np.round(stay), 10, 150)

# Borrowing: grows with stay, Premium bonus
books = (0.04 * stay + 1.6 * (plan == "Premium") + rng.normal(0, 1.2, N))
books = np.clip(np.round(books), 0, 10).astype(int)

dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, N), unit="D")

df = pd.DataFrame({
    "Member_ID": [f"LB{2001 + i}" for i in range(N)],
    "Member_Type": member_type, "Department": department,
    "Preferred_Genre": genre, "Membership_Plan": plan,
    "Preferred_Visit_Time": visit_time, "Age": age,
    "Visits_Per_Month": visits, "Avg_Stay_Minutes": stay,
    "Distance_From_Home_Km": distance,
    "Visit_Date": dates.strftime("%Y-%m-%d"),
    "Books_Borrowed_Per_Month": books,
}).sample(frac=1, random_state=7).reset_index(drop=True)

# ---- Intentional data-quality problems -------------------------------------
for col, k in [("Preferred_Genre", 6), ("Age", 6), ("Avg_Stay_Minutes", 6),
               ("Distance_From_Home_Km", 5), ("Visit_Date", 4),
               ("Preferred_Visit_Time", 5)]:
    df.loc[rng.choice(df.index, k, replace=False), col] = np.nan

df.loc[3, "Age"] = -19      # impossible ages
df.loc[17, "Age"] = 135

df.loc[5, "Department"] = " cse"            # inconsistent text
df.loc[11, "Department"] = "MECHANICAL"
df.loc[23, "Department"] = "management "
df.loc[40, "Department"] = "  Civil"
df.loc[8, "Membership_Plan"] = "premium"
df.loc[29, "Membership_Plan"] = "BASIC "

df = pd.concat([df, df.sample(6, random_state=3)], ignore_index=True)  # 6 duplicates

df.to_csv("data/raw/library_usage.csv", index=False)
df.to_excel("data/raw/library_usage.xlsx", index=False)
print("Raw dataset created:", df.shape)
