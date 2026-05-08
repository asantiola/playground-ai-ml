import sqlite3
import json
import os

conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

create_table_sql = """
    CREATE TABLE IF NOT EXISTS financial (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exchange TEXT,
        company TEXT,
        ticker TEXT,
        market_cap TEXT,
        price REAL,
        bid REAL,
        ask REAL,
        sector TEXT,
        country TEXT,
        pe_ratio REAL,
        eps REAL,
        debt_to_equity REAL,
        dividend_yield REAL
    )
"""
cursor.execute(create_table_sql)

insert_sql = """
    INSERT INTO financial (exchange, company, ticker, market_cap, price, bid, ask, sector, country, pe_ratio, eps, debt_to_equity, dividend_yield)
                   VALUES (       ?,       ?,      ?,           ?,     ?,  ?,   ?,      ?,       ?,        ?,   ?,              ?,              ?)
"""

HOME = os.environ["HOME"]
with open("data/financial01.json") as f:
    data = json.load(f)    
    for item in data["market_data"]:        
        cursor.execute(insert_sql, (item["exchange"], item["company"], item["ticker"], item["market_cap"], item["price"], item["bid"], item["ask"], item["sector"], item["country"], item["pe_ratio"], item["eps"], item["debt_to_equity"], item["debt_to_equity"]))
