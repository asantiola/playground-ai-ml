from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
import pandas as pd
import sqlite3
import json
import os

openai_base_url = os.environ.get(
    "OPENAI_BASE_URL", 
    "http://model-runner.docker.internal/engines/v1"
)

api_key = os.environ.get(
    "OPENAI_API_KEY",
    "your-default-key"
)

llm = ChatOpenAI(
    model="ai/gemma4:E4B",
    base_url=openai_base_url,
    api_key=api_key,
)

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

if __name__ == "__main__":
    # Define paths based on the task requirements
    json_file = 'data/financial01.json'
    db_file = 'data/financial01.db'
    table = 'financial_data'
    
    # # Execute the process
    # print("Starting financial data processing...")
    # success = process_and_insert_financial_data(json_file, db_file, table)
    
    # if success:
    #     print("\n--- Task Execution Complete ---")
    #     print(f"Data successfully read from {json_file} and inserted into {db_file}.")
    #     print("To verify, you can run: sqlite3 data/financial01.db")
    # else:
    #     print("\n--- Task Execution Failed ---")

    db = SQLDatabase.from_uri(f"sqlite:///{db_file}")
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description",
        verbose=False,
    )
    response = agent.invoke({"input": "Give me the top 3 performing companies, and explain why."})
    print(response["output"])
