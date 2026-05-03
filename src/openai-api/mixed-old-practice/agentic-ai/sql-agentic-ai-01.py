import os
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# practice code agentic ai
# improvements from sql-multiple-agents*.py

HOME=os.environ["HOME"]

def populate_departments(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    ''')

    cursor.execute("INSERT INTO departments VALUES (1, 'IT');")
    cursor.execute("INSERT INTO departments VALUES (2, 'HR');")
    cursor.execute("INSERT INTO departments VALUES (3, 'Marketing');")
    cursor.execute("INSERT INTO departments VALUES (4, 'Finance');")

def populate_employees(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER,
            name TEXT NOT NULL,
            salary REAL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    ''')

    cursor.execute("INSERT INTO employees VALUES(1, 1, 'Alex', 10000);")
    cursor.execute("INSERT INTO employees VALUES(2, 1, 'John', 9000);")
    cursor.execute("INSERT INTO employees VALUES(3, 1, 'Mary', 9500);")
    cursor.execute("INSERT INTO employees VALUES(4, 1, 'Joseph', 9700);")
    cursor.execute("INSERT INTO employees VALUES(5, 1, 'Jane', 9200);")
    
    cursor.execute("INSERT INTO employees VALUES(6, 2, 'Monique', 8000);")
    cursor.execute("INSERT INTO employees VALUES(7, 2, 'Owen', 8100);")

    cursor.execute("INSERT INTO employees VALUES(8, 3, 'Fred', 7200);")
    cursor.execute("INSERT INTO employees VALUES(9, 3, 'Michelle', 8100);")
    cursor.execute("INSERT INTO employees VALUES(10, 3, 'Janice', 9800);")

def populate_contacts(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            phone TEXT,
            email TEXT,
            address TEXT,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    ''')

    cursor.execute("INSERT INTO contacts VALUES(1, 2, '1234-5678', 'john@email.com', 'binondo, manila');")
    cursor.execute("INSERT INTO contacts VALUES(2, 8, '1111-2222', 'fred@email.com', 'ermita, manila');")
    cursor.execute("INSERT INTO contacts VALUES(3, 10, '3456-1234', 'janice@email.com', 'makati, manila');")
    cursor.execute("INSERT INTO contacts VALUES(4, 7, NULL, 'owen@email.com', NULL);")

def print_tables(cursor):
    cursor.execute("select * from departments")
    res = cursor.fetchall()
    print(f"departments: {res}")

    cursor.execute("select * from employees")
    res = cursor.fetchall()
    print(f"employees: {res}")

    cursor.execute("select * from contacts")
    res = cursor.fetchall()
    print(f"contacts: {res}")

    print("")

db_name = HOME + "/repo/playground-ai-ml/data/sql-agentic-ai.db"
db_type = "sqlite3"

conn = None
cursor = None
do_setup = False

try:
    do_setup = not os.path.exists(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    if do_setup:
        populate_departments(cursor)
        populate_employees(cursor)
        populate_contacts(cursor)
        conn.commit()
    
    print_tables(cursor)
except sqlite3.Error as e:
    print(f"Error caught: {e}")
    exit(1)

llm = ChatOpenAI(
    model="ai/gpt-oss:latest",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

@tool
def database_query(sql: str):
    """
    Queries a database.

    Args:
        sql (str): The SQL query to be done.
    """

    print(f"database_query: {sql}\n")
    try:
        cursor.execute(sql)
    except sqlite3.Error as e:
        print(f"database_query error: {e}")
        return None
    return cursor.fetchall()

tools = [
    database_query
]

available_functions = {
    "database_query": database_query,
}

def agent_paraphraser(input, format):
    prompt = PromptTemplate(
        template="""You are a helpful assistant that paraphrase the input into sentence form.
            Take note of the input description to help in your conversion.
            Input: {input}
            Input format: {format}
            """,
        input_variables=["input", "format",],
    )
    chain = (prompt | llm)
    response = chain.invoke({
        "input": input,
        "format": format,
    })
    return response.content

def agent_tableinfo():
    format = "[table1, (column1, column2, ...), table2, (column1, column2, ...), ...]"
    prompt = PromptTemplate(
        template="""You are a helpful assistant, with expertise in {db_type} database.
            Given a database {db_name}, generate a SQL that can be called with the tool.
            The SQL query provide me information in the following format:
            {format}
            """,
        input_variables=["db_type", "db_name", "format"],
    )
    chain = (prompt | llm.bind_tools([database_query]))
    response = chain.invoke({
        "db_type": db_type,
        "db_name": db_name,
        "format": format,
    })
    # print(f"response: {response}")

    info = ""
    for tool in response.tool_calls:
        if function_to_call := available_functions.get(tool['name']):
            # print(f"calling {tool['name']}")
            tool_response = function_to_call.invoke(tool['args'])
            # print(f"tool_response: {tool_response}")
            info += " " + agent_paraphraser(input=tool_response, format=format)
    # print(f"info: {info}")
    return info

db_info = agent_tableinfo()
# print(f"tables and columns: {db_info}")

def agent_expert(question):
    prompt = PromptTemplate(
        template="""You are a helpful assistant that helps answer questions.
            Assess the given question.
            If the database has enough to answer it, execute the tool with a SQL to get the answer.
            You can use joins and unions in your SQL.
            Otherwise, you do not have to call the tool, you can answer the question based on your training data.
            Question: {question}
            Database information: {db_info}
            """,
        input_variables=["question", "db_info",],
    )
    chain = (prompt | llm.bind_tools([database_query]))
    response = chain.invoke({
        "question": question,
        "db_info": db_info,
    })

    format = f"The input is SQL query output to the question {question}."
    info = ""
    for tool in response.tool_calls:
        if function_to_call := available_functions.get(tool['name']):
            # print(f"calling {tool['name']}")
            tool_response = function_to_call.invoke(tool['args'])
            # print(f"tool_response: {tool_response}")
            info += " " + agent_paraphraser(input=tool_response, format=format)
    # print(f"info: {info}")

    if info == "":
        info = response.content
    return info

def ask_alex(question):
    answer = agent_expert(question)
    print(f"question: {question}\nanswer: {answer}\n")

ask_alex("Who is the employee with the highest salary, and how much was the salary?")
ask_alex("Who are the top three highest earning employees, and what are their salaries?")
ask_alex("What is the largest bone in the human body?")
ask_alex("Which department has the most number of employees, and how many employees does it have?")
ask_alex("Among the employees with contact information, who has the highest salary and what is the email address?.")
ask_alex("How many moons does Jupiter have?")

conn.close()

# # results using gpt-oss

# departments: [(1, 'IT'), (2, 'HR'), (3, 'Marketing'), (4, 'Finance')]
# employees: [(1, 1, 'Alex', 10000.0), (2, 1, 'John', 9000.0), (3, 1, 'Mary', 9500.0), (4, 1, 'Joseph', 9700.0), (5, 1, 'Jane', 9200.0), (6, 2, 'Monique', 8000.0), (7, 2, 'Owen', 8100.0), (8, 3, 'Fred', 7200.0), (9, 3, 'Michelle', 8100.0), (10, 3, 'Janice', 9800.0)]
# contacts: [(1, 2, '1234-5678', 'john@email.com', 'binondo, manila'), (2, 8, '1111-2222', 'fred@email.com', 'ermita, manila'), (3, 10, '3456-1234', 'janice@email.com', 'makati, manila'), (4, 7, None, 'owen@email.com', None)]

# database_query: SELECT '[' || group_concat(name || ', (' || cols || ')', ', ') || ']' AS result FROM (SELECT name, (SELECT group_concat(name, ', ') FROM pragma_table_info(t.name)) AS cols FROM sqlite_master t WHERE type='table' ORDER BY name)

# database_query: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1

# question: Who is the employee with the highest salary, and how much was the salary?
# answer:  Alex earns the highest salary, which is $10,000.

# database_query: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3

# question: Who are the top three highest earning employees, and what are their salaries?
# answer:  The top three highest‑earning employees are Alex, who earns $10,000; Janice, who earns $9,800; and Joseph, who earns $9,700.

# question: What is the largest bone in the human body?
# answer: The largest bone in the human body is the **femur** (thigh bone).

# database_query: SELECT d.name, COUNT(e.id) AS employee_count FROM departments d JOIN employees e ON d.id = e.department_id GROUP BY d.id ORDER BY employee_count DESC LIMIT 1;

# question: Which department has the most number of employees, and how many employees does it have?
# answer:  The IT department has the most employees, with a total of five.

# database_query: SELECT e.name, c.email FROM employees e JOIN contacts c ON e.id = c.employee_id ORDER BY e.salary DESC LIMIT 1;

# question: Among the employees with contact information, who has the highest salary and what is the email address?.
# answer:  Janice, whose email address is janice@email.com, has the highest salary among employees with contact information.

# question: How many moons does Jupiter have?
# answer: Jupiter has **79 known moons** (as of the most recent counts in 2023).
