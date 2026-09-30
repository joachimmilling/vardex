"""One question, asked three ways: plain Python, pandas and DuckDB.

Run with:  uv run --with pandas --with duckdb python examples/same_question.py
The companies are made up.
"""

from collections import defaultdict

import duckdb
import pandas as pd

rows = [
    {"name": "Demo Laks AS", "county": "Vestland", "industry": "Fish farming", "employees": 240},
    {"name": "Demo Sjømat AS", "county": "Nordland", "industry": "Fish farming", "employees": 130},
    {"name": "Demo Bygg AS", "county": "Vestland", "industry": "Construction", "employees": 45},
    {"name": "Demo Data AS", "county": "Oslo", "industry": "Consulting", "employees": 60},
    {"name": "Demo Transport AS", "county": "Trøndelag", "industry": "Transport", "employees": 85},
    {"name": "Demo Fiske AS", "county": "Vestland", "industry": "Fishing", "employees": 12},
    {"name": "Demo Havbruk AS", "county": "Vestland", "industry": "Fish farming", "employees": 95},
]

# The question: employees per county, counting only companies with more than 50 employees,
# largest county first.

# 1. Plain Python
totals: dict[str, int] = defaultdict(int)
for row in rows:
    if row["employees"] > 50:  # WHERE
        totals[row["county"]] += row["employees"]  # GROUP BY county, SUM(employees)
python_result = sorted(totals.items(), key=lambda item: item[1], reverse=True)  # ORDER BY
print("Plain Python:", python_result)

# 2. pandas
df = pd.DataFrame(rows)
pandas_result = (
    df[df["employees"] > 50]
    .groupby("county", as_index=False)["employees"]
    .sum()
    .sort_values("employees", ascending=False)
)
print("pandas:\n", pandas_result)

# 3. DuckDB, running SQL directly on the pandas DataFrame called df
duckdb_result = duckdb.sql(
    """
    SELECT county, SUM(employees)::BIGINT AS employees  -- see "Types change at borders"
    FROM df
    WHERE employees > 50
    GROUP BY county
    ORDER BY employees DESC
    """
).df()
print("DuckDB:\n", duckdb_result)
