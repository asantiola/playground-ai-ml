import os
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

HOME=os.environ["HOME"]
db_name = HOME + "/repo/playground-ai-ml/data/sql-transactions.db"
db_uri = "sqlite:///" + HOME + "/repo/playground-ai-ml/data/sql-transactions.db"

api_url = "http://localhost:12434/engines/v1"
api_key = "docker"
llm_model = "ai/gpt-oss:latest"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    base_url=api_url,
    api_key=api_key,
)

def setup_db():
    if os.path.exists(db_name):
        os.remove(db_name)
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant expert in sqlite database.
        You can use unions and joins if required.
        If preparing a DML, prepare a single statement.
        No need to enclose the SQL in quotes.
        No need to provide an explanation for your answer.
        Generate a SQL for the given input: {input}
        """
    )
    chain = prompt | llm | StrOutputParser()

    sql = chain.invoke({
        "input": """
            Create a table called transactions with the following columns:
            'id' is the primary key, it is an autoincrementing integer.
            'code' is a string of length 2, cannot be null. This is the transaction code.
            'currency' is an integer. This is ISO 4217 numeric code for the currency.
            'amount' is a real number representing amount. 
        """
    })
    print(f"sql:\n{sql}\n")
    cursor.execute(sql)
    
    sql = chain.invoke({
        "input": """
            Add the following rows into 'transactions':
            Code: '01', currency: 840, amount: 12.34
            Code: '02', currency: 840, amount: 23.45
            Code: '02', currency: 840, amount: 34.56
            Code: '03', currency: 840, amount: 45.67
            Code: '03', currency: 840, amount: 56.78
            Code: '03', currency: 702, amount: 67.89
            Code: '04', currency: 702, amount: 78.90
            Code: '04', currency: 702, amount: 89.01
            Code: '04', currency: 702, amount: 90.12
            Code: '04', currency: 608, amount: 98.76
            Code: '05', currency: 608, amount: 87.65
            Code: '05', currency: 608, amount: 76.54
            Code: '05', currency: 978, amount: 65.43
            Code: '05', currency: 978, amount: 54.32
            Code: '05', currency: 124, amount: 43.21
        """
    })
    print(f"sql:\n{sql}\n")
    cursor.execute(sql)
    conn.commit()

# setup_db()

db = SQLDatabase.from_uri(db_uri)
sql_agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=False)

response = sql_agent.invoke({ 
    "input": """
        Group the transactions by 'code' and 'currency', then give me count and sum of amounts.
        Order the result by code and currency.
        The output columns are: 
        - 'code'
        - 'filler1', 2 blank spaces.
        - 'currency'
        - 'filler2', 1 blank space.
        - 'count', should be reformatted to string of length 8, zero-prefixed.
        - 'amount', should be reformatted to a string of length 12, zero-prefixed. Remove the decimal point it will be implied.        
        Output in JSON format.
    """ 
})
print(f"input:\n{response["input"]}\n")
print(f"output:\n{response["output"]}\n")

# Alternative output:
# Output the values only in fixed-length text format. Do NOT add extra space between output columns.
# No need for quotes or explanations.
