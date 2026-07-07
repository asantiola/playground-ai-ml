from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated
from langchain_core.tools import tool
import pandas as pd
import sqlite3
import json
import os

workspaces = os.environ.get(
    "WORKSPACES",
    "/workspaces"
)

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://localhost:12434/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="mlx-community/gemma-4-12B-it-qat-6bit",
    base_url=openai_base_url,
    api_key=api_key,
)

json_file = 'data/financial01.json'
db_file = 'data/financial01.db'
table = 'financial_data'

@tool
def list_tables() -> str:
    """Returns a list of tables available in the database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return f"Tables: {', '.join(tables)}"

@tool
def get_schema(table_name: str) -> str:
    """Returns the column names and types for a specific table."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    conn.close()
    return str(schema)

@tool
def run_sql_query(query: Annotated[str, "A valid SQLite SELECT query"]) -> str:
    """Executes a SQL query and returns the results as a string."""
    # Safety Check: Only allow SELECT statements
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for security."
    
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_string()
    except Exception as e:
        return f"Error: {str(e)}"

tools = [list_tables, get_schema, run_sql_query]
llm_with_tools = llm.bind_tools(tools)

def process_and_insert_financial_data(json_path, db_path, table_name="financial_data"):
    """
    Reads financial data from a JSON file and inserts it into an SQLite database.
    Handles file paths relative to the current working directory.
    """
    # Ensure the data directory exists before proceeding
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        # 1. Read JSON data using built-in json module and pandas
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data.get("market_data", []))
        print(f"Successfully read {len(df)} records from {json_path}")

        # 2. Insert into SQLite database
        conn = sqlite3.connect(db_path)
        # If the table exists, it will be replaced.
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Successfully inserted data into table '{table_name}' in {db_path}")
        return True
    except FileNotFoundError as e:
        print(f"Error: File not found at {e.filename}")
        return False
    except Exception as e:
        print(f"An error occurred during data processing or database insertion: {e}")
        return False

def query_analyst(user_question: str):
    system_prompt = """
    You are a SQL analyst. First, list tables.
    Then get the schema.
    Finally, write a query to answer the user.
    Limit results to 10.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]
    
    # First call: LLM decides which tool to use
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Tool Execution Loop
    while ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            # Map tool call to the actual function
            tool_name = tool_call["name"].lower()
            tool_func = {
                "list_tables": list_tables,
                "get_schema": get_schema,
                "run_sql_query": run_sql_query
            }[tool_name]
            
            # Execute tool and add to message history
            output = tool_func.invoke(tool_call["args"])
            messages.append({
                "role": "tool",
                "content": str(output),
                "tool_call_id": tool_call["id"]
            })
        
        # Ask LLM again with the tool results
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

    return ai_msg.content

if __name__ == "__main__":
    # # Execute the process
    # print("Starting financial data processing...")
    # success = process_and_insert_financial_data(json_file, db_file, table)
    
    # if success:
    #     print("\n--- Task Execution Complete ---")
    #     print(f"Data successfully read from {json_file} and inserted into {db_file}.")
    #     print("To verify, you can run: sqlite3 data/financial01.db")
    # else:
    #     print("\n--- Task Execution Failed ---")

    result = query_analyst("Give me the top 3 performing companies, and explain why.")
    print(f"\nFinal Answer:\n{result}")

    result = query_analyst("Based on the information in the database for Visa Inc. vs Mastercard Inc., which is better to buy?")
    print(f"\nFinal Answer:\n{result}")
