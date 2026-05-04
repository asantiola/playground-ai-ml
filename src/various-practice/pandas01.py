from operator import contains

from numpy import empty
import pandas as pd
import json
import os

HOME = os.environ["HOME"]
financial_file = HOME + "/repo/playground-ai-ml/data/financial01.json"

try:
    with open(financial_file, "r") as f:
        financial_data = json.load(f)
except FileNotFoundError:
    print(f"Error: file not found: {financial_file}")
    exit
except json.JSONDecodeError:
    print(f"Error: file {financial_file} is not a valid JSON")
    exit

df = pd.DataFrame(financial_data["market_data"])
df["div_val"] = df["dividend_yield"].str.replace("%", "").astype(float)

print(df)

found = df.query("company.str.contains('inc', case=False)")
print(f"FOUND: {found}")

# 2. Normalize Metrics (0 to 1 scale)
# For PE and Debt: Lower is better, so we subtract from 1
df['pe_score'] = 1 - (df['pe_ratio'] - df['pe_ratio'].min()) / (df['pe_ratio'].max() - df['pe_ratio'].min())
df['debt_score'] = 1 - (df['debt_to_equity'] - df['debt_to_equity'].min()) / (df['debt_to_equity'].max() - df['debt_to_equity'].min())
df['eps_score'] = (df['eps'] - df['eps'].min()) / (df['eps'].max() - df['eps'].min())
df['yield_score'] = (df['div_val'] - df['div_val'].min()) / (df['div_val'].max() - df['div_val'].min())

# 3. Calculate Final Weighted Score (Equal 25% weights)
df['final_score'] = (df['pe_score'] + df['debt_score'] + df['eps_score'] + df['yield_score']) / 4 * 100
top_picks = df.sort_values(by='final_score', ascending=False)
print(top_picks[['company', 'ticker', 'final_score']].head(3))

print(df[df["sector"] == "Information Technology"])

print(df["company"])
